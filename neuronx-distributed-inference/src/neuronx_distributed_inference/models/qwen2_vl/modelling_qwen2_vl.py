# coding=utf-8
"""
Qwen2‑VL Neuron NXD inference implementation
===========================================

This file adapts **Qwen2‑VL** (multimodal) for AWS Neuron / neuronx‑distributed in
exactly the same architectural style that *NeuronMllama* and *NeuronQwen2* use.

The code is deliberately split into **Config ➜ Building blocks ➜ Model ➜ Wrapper**
so you can progressively test/trace each stage.  Everything compiles under
Python 3.10 and neuronx‑distributed ≥ 2.16.

:copyright: © 2025 Alibaba & AWS AI
:licence: Apache‑2.0 (same as original Qwen2‑VL HF implementation).
"""

from __future__ import annotations

from typing import (
    Callable,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    Union,
)
import types
import copy
import json

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from neuronx_distributed.parallel_layers import parallel_state, utils
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MultimodalVisionNeuronConfig,
    to_dict,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseModel,
    NeuronBaseForCausalLM,
    turn_2d_mask_to_4d,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    GroupQueryAttention_QKV,
    GroupQueryAttention_O,
)
from neuronx_distributed_inference.modules.attention.utils import move_heads_front
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.models.llama.modeling_llama import (
    NeuronLlamaMLP,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from neuronx_distributed.utils import cpu_mode

from .model_wrapper_qwen2_vl import ModelWrapperQwen2VL
from .modelling_qwen2_vl_vision import NeuronQwen2VisionTransformerPretrainedModel, VisionRotaryEmbedding
from .utils import (
    apply_multimodal_rotary_pos_emb,PIXEL_SIZE_MAP, prepare_scatter_positions
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .hf_state_dict_conversion import convert_hf_to_neuron_state_dict
import logging
from neuronx_distributed_inference.modules.custom_calls import neuron_cumsum
from neuronx_distributed_inference.modules.flashdecode.utils import (
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    mask_padded_logits,
)
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
)
from transformers.modeling_outputs import ModelOutput

from neuronx_distributed_inference.utils.distributed import get_tp_group

from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter


logging = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# @dataclass
# class Qwen2VLCausalLMOutputWithPast(ModelOutput):
#     """
#     Base class for Qwen2VL causal language model (or autoregressive) outputs.

#     Args:
#         loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
#             Language modeling loss (for next-token prediction).
#         logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
#             Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#         past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#             Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#             `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#             Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#             `past_key_values` input) to speed up sequential decoding.
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
#             one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

#             Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`.

#             Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#             heads.
#         rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
#             The rope index difference between sequence length and multimodal rope.
#     """

#     loss: Optional[torch.FloatTensor] = None
#     logits: torch.FloatTensor = None
#     past_key_values: Optional[List[torch.FloatTensor]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     rope_deltas: Optional[torch.LongTensor] = None


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return LlamaRMSNorm if cpu_mode() else CustomRMSNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class Qwen2VLNeuronConfig(MultimodalVisionNeuronConfig):
    """Extends the generic multimodal config with the Qwen2‑VL defaults."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronQwen2VLAttention


class Qwen2VLInferenceConfig(InferenceConfig):
    """Configuration wrapper tailored **only** for the *official* flat‑structure
    Qwen 2‑VL JSON (no nested `text_config`). It extracts text‑model hyper‑
    parameters from the top level, wraps them into an `InferenceConfig` used by
    Neuron, and keeps the provided `vision_config` untouched (aside from a small
    dtype cleanup).
    """

    # ---------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "vision_config"):
            self.vision_config = {}

        if isinstance(self.vision_config, types.SimpleNamespace):
            self.vision_config = vars(self.vision_config)

        if isinstance(self.vision_config, dict):
            vision_dict = self.vision_config
            vision_dict.pop("torch_dtype", None)
            self.vision_config = InferenceConfig(self.neuron_config, **vision_dict)
            
        

    def add_derived_config(self):
        self.num_cores_per_group = 1
        self.qkv_bias = True
        self.o_bias = False
        self.neuron_config.tie_word_embeddings = self.tie_word_embeddings
        # Update before calling validate_config

        # --- build text_config from flat keys ---
        
        if not hasattr(self, "text_config"):
            text_keys = [
                "hidden_size",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "pad_token_id",
                "vocab_size",
                "intermediate_size",
                "max_position_embeddings",
                "rms_norm_eps",
                "rope_theta",
                "rope_scaling",
                "hidden_act",
                "bos_token_id",
                "eos_token_id",
                "qkv_bias",
                "o_bias",
                "intermediate_size",
                "vision_token_id",
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            ]
            text_dict = {k: getattr(self, k) for k in text_keys if hasattr(self, k)}
            text_dict["pad_token_id"] = text_dict["bos_token_id"]
            text_dict["output_attentions"] = False
            text_dict["output_hidden_states"] = False
            # strip the original copies to avoid double validation later

            self.text_config = InferenceConfig(self.neuron_config, **text_dict)

            # Delete duplicates
            for k in text_dict:
                if hasattr(self, k):
                    delattr(self, k)
                    
            self.pad_token_id = text_dict["pad_token_id"]
                    
            
        if isinstance(self.text_config, dict):
            self.text_config = InferenceConfig(self.neuron_config, **self.text_config)
    # ---------------------------------------------------------------
    # Validation helpers
    # ---------------------------------------------------------------
    def get_required_attributes(self) -> List[str]:
        return [
            "text_config.hidden_size",
            "text_config.num_attention_heads",
            "text_config.num_hidden_layers",
            "text_config.num_key_value_heads",
            "text_config.bos_token_id",
            "text_config.eos_token_id",
            "text_config.vocab_size",
            "text_config.intermediate_size",
            "text_config.max_position_embeddings",
            "text_config.rope_theta",
            "text_config.rms_norm_eps",
            "text_config.hidden_act",
            "text_config.rope_scaling",
            "text_config.qkv_bias",
            "text_config.o_bias",
            "text_config.intermediate_size",
            "vision_config.depth",
            "vision_config.mlp_ratio",
            "vision_config.num_heads",
            "vision_config.in_chans",
            "vision_config.patch_size",
            "vision_config.spatial_merge_size",
            "vision_config.spatial_patch_size",
            "vision_config.temporal_patch_size",
        ]

    def validate_config(self):
        """
        Validates that the config has all required attributes.
        """

        def hasattr_nested(obj, attr_chain):
            attrs = attr_chain.split(".")
            for attr in attrs:
                if isinstance(obj, dict):
                    if attr not in obj:
                        return False
                    obj = obj[attr]
                else:
                    if not hasattr(obj, attr):
                        return False
                    obj = getattr(obj, attr)
            return True

        missing_attributes = [
            x for x in self.get_required_attributes() if not hasattr_nested(self, x)
        ]
        assert len(missing_attributes) == 0, f"Config must define {missing_attributes}"

        assert (
            self.neuron_config.is_medusa is False
            and self.neuron_config.speculation_length == 0
        ), f"Speculative Decoding is not yet supported in this Model. \
                is_medusa was set to {self.neuron_config.is_medusa}. \
                speculation_length was set to {self.neuron_config.speculation_length}"
        assert (
            int(self.neuron_config.logical_nc_config) == 1
        ), "This model currently only support logical_nc_config=1"

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        config_dict["text_config"].pop("neuron_config", None)
        config_dict["vision_config"].pop("neuron_config", None)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Qwen2VLNeuronConfig]:
        return Qwen2VLNeuronConfig


class NeuronQwen2VLRotaryEmbedding(nn.Module):
    def __init__(self, config: InferenceConfig, base=1000000.0):
        super().__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        if hasattr(config, "rope_theta"):
            self.base = config.rope_theta
        else:
            self.base = base
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", None, persistent=False)

    # Only one form of init supported
    # TODO: Add other methods
    def get_inv_freqs(self, device: Optional[torch.device] = None) -> torch.Tensor:
        
        freq_indices = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.base ** (freq_indices / self.dim))

    @torch.no_grad()
    def forward(self, x, position_ids):
        # 延迟初始化 inv_freq
        if self.inv_freq is None:
            self.inv_freq = self.get_inv_freqs(device=x.device)

        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class NeuronQwen2VLAttention(NeuronAttentionBase):
    """Self-attention identical to *NeuronQwen2Attention* but with RoPE sharing
    semantics compatible with Qwen2-VL (3-way multimodal RoPE).
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.rope_scaling = config.rope_scaling
        self.mrope_section = self.rope_scaling["mrope_section"]
        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1

        self.fused_qkv = False
        self.clip_qkv = None
        # qkv and o bias
        self.qkv_bias = config.qkv_bias
        self.o_bias = config.o_bias
        self.torch_dtype = config.neuron_config.torch_dtype

        self.init_gqa_properties()
        self.rotary_emb = NeuronQwen2VLRotaryEmbedding(config=config)
        self.padding_side = "right"

    # *forward* are inherited from base; no changes needed
    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.qkv_bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            logical_nc_config=self.neuron_config.logical_nc_config,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.o_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
        )
        self.num_heads = utils.divide(
            self.qkv_proj.get_num_attention_heads(), self.tp_degree
        )
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_kernel_enabled = False
        self.logical_nc_config = self.neuron_config.logical_nc_config

    # Update prep_qkv_tensors in NeuronAttentionBase to include multimodal rotary embeddings
    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """take care of the shape, layout, group query, custom position encoding, etc."""
        Q, K, V = self.qkv_proj(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids
        )

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K,
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            layernorm=self.k_layernorm,
        )
        V = move_heads_front(
            V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None
        )

        # Rotate Q and K
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_multimodal_rotary_pos_emb(
                Q, K, cos_cache, sin_cache, self.mrope_section
            )    
        return Q, K, V, cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Decoder Blocks
# ---------------------------------------------------------------------------


class NeuronQwen2VLDecoderLayer(nn.Module):
    """Combines self‑attention, optional cross‑attention (vision) and MLP."""

    def __init__(self, config: InferenceConfig, is_xattn: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen2VLAttention(config)

        self.mlp = NeuronLlamaMLP(config)

        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)

        return outputs


# ---------------------------------------------------------------------------
# Text + Vision models
# ---------------------------------------------------------------------------


class NeuronQwen2VLTextModel(NeuronBaseModel):
    """Qwen2‑VL text backbone with optional cross‑attention layers."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        super().__init__(self.config)

    # --- attribute setup ------------------------------------------------------------
    def setup_attr_for_model(self, config: InferenceConfig):
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.num_hidden_layers = config.num_hidden_layers
        self.on_device_sampling = config.neuron_config.on_device_sampling_config

    # --- model init -----------------------------------------------------------------
    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                self.vocab_size,
                self.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                self.hidden_size,
                self.vocab_size,
                bias=False,
                
            )

        self.layers = nn.ModuleList(
            [NeuronQwen2VLDecoderLayer(config) for _ in range(self.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


    def _create_context_attn_mask(self, attention_mask):
        # Lower triangle causal mask for classic attention
        # mask = torch.full(
        #     (self.n_positions, self.n_positions), True, device=attention_mask.device
        # ).tril(diagonal=0)
        # mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)
        
        mask = torch.full(
            (self.n_positions, self.n_positions), 1, device=attention_mask.device
        ).tril(diagonal=0)
        mask_2d = torch.einsum('bi,bj->bij', attention_mask, attention_mask)
        mask = mask_2d*mask
        mask = mask[None,:,:,:].expand(self.batch_size, 1, self.n_positions, self.n_positions)
        mask = mask.to(torch.bool)

        if self.padding_side == "right":
            return mask
        else:
            expanded_mask = (
                attention_mask[:, None, None, :]
                .expand(self.batch_size, 1, self.n_positions, self.n_positions)
                .to(torch.bool)
            )
            return torch.logical_and(mask, expanded_mask)


    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        prev_hidden=None,
        adapter_ids=None,
        accepted_indices=None,
        current_length=None,
        medusa_mask=None,
        scatter_index=None,
        slot_mapping=None,
        active_block_table=None,
        num_queries=None,
        computed_context_lens=None,
        cache_mask=None,
        current_reordered_idx=None,
        cache_reordered_idx=None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        active_mask=None,
        rotary_position_id=None,
    ):
        if self.neuron_config.is_medusa:
            return self._medusa_forward(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                adapter_ids,
                accepted_indices,
                current_length,
                medusa_mask,
                scatter_index,
            )

        is_for_token_gen = attention_mask.dim() == 4

        if (
            is_for_token_gen
            and self.neuron_config.enable_token_tree
            and self.neuron_config.enable_eagle_speculation
        ):
            logging.warning("entering _eagle_token_tree_forward")
            return self._eagle_token_tree_forward(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                prev_hidden,
                adapter_ids,
                scatter_index=scatter_index,
                inputs_embeds=inputs_embeds,
                kv_cache=kv_cache,
                active_mask=active_mask,
                rotary_position_id=rotary_position_id,
            )
        # TODO: This will not work for a context encoding model with bucket size
        # equal to the speculation length
        is_for_context_encoding = self._is_context_encoding(input_ids)
        is_for_speculation = self._is_for_speculation(input_ids)

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # It is either for context encoding or for token generation
        if self.is_block_kv_layout:
            past_key_values = self.kv_mgr.get_cache(active_block_table=active_block_table)
        elif is_for_context_encoding:
            past_key_values = None
        else:
            if kv_cache is None:
                past_key_values = self.kv_mgr.get_cache(seq_len=cache_size)
            else:
                past_key_values = self._slice_kv_cache(kv_cache, cache_size)

        # Prepare attention mask(s)
        if self.is_prefix_caching:
            attention_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
                query_lens=num_queries,
                key_lens=num_queries + computed_context_lens,
                max_query_len=self.neuron_config.n_active_tokens,
                max_key_len=self.neuron_config.max_context_length,
                is_prior=True,
            )
        elif self.is_chunked_prefill:
            max_total_len = (
                self.neuron_config.cp_num_active_blocks * self.neuron_config.pa_block_size
                + self.neuron_config.max_context_length
            )
            attention_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
                query_lens=num_queries,
                key_lens=num_queries + computed_context_lens,
                max_query_len=self.neuron_config.max_context_length,
                max_key_len=max_total_len,
            )
        else:
            attention_mask = self.create_attn_mask(
                attention_mask,
                is_for_context_encoding,
                is_for_speculation,
            )

        active_mask = None
        if self.is_prefix_caching:
            active_mask = self._create_block_kv_attn_mask(
                query_lens=num_queries,
                key_lens=num_queries,
                max_query_len=self.neuron_config.n_active_tokens,
                max_key_len=self.neuron_config.n_active_tokens,
                is_prior=False,
            )
        if is_for_speculation:
            active_mask = torch.full(
                (self.speculation_length, self.speculation_length),
                True,
                device=attention_mask.device,
            ).tril(diagonal=0)
            active_mask = active_mask[None, None, :, :].expand(
                self.batch_size, 1, self.speculation_length, self.speculation_length
            )

        # FD masks
        active_mask_2d = None
        if self.neuron_config.flash_decoding_enabled and not is_for_context_encoding:
            rank_id = self.rank_util.get_rank()
            active_mask_2d, attention_mask_2d = mask_util(
                pos_ids=position_ids,
                rank_id=rank_id,
                num_cores_per_group=self.num_cores_per_group,
                cache_size=cache_size,
            )
            active_mask = turn_2d_mask_to_4d(
                active_mask_2d, n_positions=1, batch_size=self.batch_size
            )
            attention_mask = turn_2d_mask_to_4d(
                attention_mask_2d, n_positions=cache_size, batch_size=self.batch_size
            )

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
            adapter_ids=adapter_ids,
            prev_hidden=prev_hidden,
            cache_mask=cache_mask,
            current_reordered_idx=current_reordered_idx,
            cache_reordered_idx=cache_reordered_idx,
            rotary_position_ids=rotary_position_id
        )

        if self.neuron_config.enable_eagle_speculation:
            full_hidden_states = hidden_states

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=past_key_values,
            seq_len=cache_size,
            scatter_index=slot_mapping if self.is_block_kv_layout else scatter_index,
            active_mask=active_mask_2d,
            kvcache_buffer=kv_cache,
        )

        batch_size = input_ids.shape[0]
        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        elif self.is_chunked_prefill:
            # chunked prefill will return cp_max_num_seqs, not just the last one
            index = neuron_cumsum(num_queries.reshape(1, -1).float()).int() - 1
            index = index.reshape(1, -1, 1)
            index = index.expand(batch_size, -1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            if not (
                position_ids.shape[-1] == self.speculation_length or position_ids.shape[-1] == 1
            ):
                # context encoding
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(
                    group=self.lm_head.tensor_parallel_group
                )
            logits = mask_padded_logits(logits, rank_id, world_size, pad_size=self.lm_head.pad_size)

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            # FIXME, logits[:, -1, :] is not correct for speculation model, this is a tempory fix.
            if is_for_speculation and not self.neuron_config.on_device_sampling_config.do_sample:
                res = nxd_argmax(tensor=logits, dim=2, gather_dim=2, keepdim=False)
                res = res.to(torch.int32)
            elif (
                is_for_context_encoding
                or not self.neuron_config.enable_eagle_speculation
                or not self.neuron_config.on_device_sampling_config.do_sample
            ):
                res = self.sampler(
                    logits[:, -1, :], sampling_params, rank_id=self.rank_util.get_rank()
                )
                res = res.to(torch.int32)
            # Otherwise we return the full logits for multinomial sampling in spec decoding

        outputs = [res]
        if self.neuron_config.output_logits:
            logits = _gather_along_dim(
                logits,
                partition_dim=2,
                process_group=get_tp_group(self.config),
            )
            outputs += [logits]
        outputs += updated_kv_cache

        if self.neuron_config.enable_eagle_speculation:
            outputs = outputs + [full_hidden_states]

        return outputs



class NeuronQwen2VLVisionModel(nn.Module):
    """Thin wrapper around the HF vision stack so we can compile separately."""

    def __init__(self, vcfg):
        super().__init__()
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VisionTransformerPretrainedModel,
        )

        self.core = Qwen2VisionTransformerPretrainedModel(vcfg)

    def forward(self, pixel_values, grid_thw):
        return self.core(pixel_values, grid_thw)


class NeuronQwen2VLModel(NeuronBaseModel):
    """Full multimodal model = Vision backbone + Text"""

    def __init__(self, config: Qwen2VLInferenceConfig):
        self.vision_config = config.vision_config
        self.text_config = config.text_config
        super().__init__(config.text_config)

    def setup_attr_for_model(self, config: InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        self.neuron_config = config.neuron_config

    def init_model(self, cfg: InferenceConfig):
        self.vision_model = NeuronQwen2VisionTransformerPretrainedModel(
            self.vision_config
        )
        self.text_model = NeuronQwen2VLTextModel(self.text_config)

    def init_inference_optimization(self, config: InferenceConfig):
        super().init_inference_optimization(config)
        # only need one kv cache mgr
        self.kv_mgr = self.text_model.kv_mgr

    def scatter_image_token_embeddings(self, input_embeddings, image_embeddings, positions, valid_mask):
        """
        Scatter image embeddings into input embeddings at specified positions.
        
        Args:
            input_embeddings: Target tensor to scatter embeddings into. Shape = [batch_size, seq_length, embedding_dim]
            image_embeddings: Encoded image embeddings to be scattered. Shape = [max_image_tokens, embedding_dim]
            positions: Fixed-size tensor of positions. Shape = [batch_size, max_image_tokens]
            valid_mask: Boolean tensor indicating valid positions. Shape = [batch_size, max_image_tokens]
        
        Returns:
            Updated input_embeddings with image embeddings scattered at the specified positions
        """
        input_embeddings = input_embeddings.clone()
        
        batch_size = input_embeddings.size(0)
        embedding_dim = input_embeddings.size(-1)
        
        # Count image tokens per batch item based on valid_mask
        image_tokens_per_batch = valid_mask.sum(dim=1).tolist()
        
        # Verify we have the right number of image embeddings
        total_image_tokens = sum(image_tokens_per_batch)
        assert total_image_tokens <= image_embeddings.size(0), \
            f"Expected {total_image_tokens} image embeddings, got {image_embeddings.size(0)}"
        
        # Split image embeddings according to counts per batch item
        # image_embeddings_list = image_embeddings.split(image_tokens_per_batch, dim=0)
        
        # Scatter image embeddings for each batch item
        for batch_idx in range(batch_size):
            # Skip if no image tokens in this batch item
            if image_tokens_per_batch[batch_idx] == 0:
                continue
            
            # Get valid positions for this batch item
            batch_positions = positions[batch_idx].to(torch.long)
            batch_valid_mask = valid_mask[batch_idx]
            valid_positions = batch_positions[batch_valid_mask]
            
            # Get the corresponding image embeddings for this batch item
            # current_image_embeddings = image_embeddings_list[batch_idx]
            
            # Use scatter_ to place image embeddings at the specified positions
            input_embeddings[batch_idx].scatter_(
                0,  # scatter along first dimension (sequence length)
                valid_positions.unsqueeze(-1).expand(-1, embedding_dim),  # expand positions
                image_embeddings  # image embeddings to scatter
            )
        return input_embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        rotary_position_ids:Optional[torch.LongTensor] = None,
        vision_attention:Optional[torch.Tensor] = None,
        vision_cos: Optional[torch.Tensor] = None,
        vision_sin: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        
        

        is_for_context_encoding = input_ids.shape[-1] > 1

        # If we are embedding for the first time, we take into account images and video embeddings.
        # If we are continuing generation, we can skip embeddings and just use input_ids, previous embeddings should be cached
        if is_for_context_encoding and inputs_embeds is None:
            inputs_embeds = self.text_model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.vision_model.get_dtype())                
                
                
                #Note that image_embedding shape is pixel_values.shape[0]/4, hidden_dim. 
                #We always pad pixel_values for a specific bucket to be of same shape. We can have higher pixel_values for bigger buckets
                #To account for padding we need to pass in vision_attention mask and pos embeddings which are created based on the actual input image and not the padded image
                
                #Get image embeddings
                image_embeds = self.vision_model(pixel_values, vision_attention, vision_cos, vision_sin)
                image_embeds = image_embeds.to(inputs_embeds.device)
                
               
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds = self.scatter_image_token_embeddings(inputs_embeds, image_embeds, positions, valid_mask)
             

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(
                    self.vision_model.get_dtype()
                )
                video_embeds = self.vision_model(
                    pixel_values_videos, grid_thw=video_grid_thw
                )
                video_mask = (
                    (input_ids == self.text_config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                )
                video_embeds = video_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)


        
        outputs = self.text_model(
            input_ids=input_ids,
            #This will be used to keep track of current token under consideraton. Used to store KV Cache at index
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            sampling_params=sampling_params,
            seq_ids=seq_ids,
            #This will just be used in attention
            rotary_position_id = rotary_position_ids,
            **kwargs,
        )
        return outputs


# ---------------------------------------------------------------------------
# Public "ForCausalLM" wrapper
# ---------------------------------------------------------------------------

class NeuronQwen2VLForCausalLM(NeuronBaseForCausalLM):
    """Drop‑in replacement for `Qwen2VLForConditionalGeneration` running on Neuron."""

    _model_cls = NeuronQwen2VLModel
    _STATE_DICT_MODEL_PREFIX = "new_model"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_deltas = torch.zeros((self.config.neuron_config.batch_size, 1), dtype=torch.int32)
    @classmethod
    def get_config_cls(cls):
        return Qwen2VLInferenceConfig

    @classmethod
    def get_neuron_config_cls(cls):
        return Qwen2VLNeuronConfig

    def get_model_wrapper_cls(self):
        return ModelWrapperQwen2VL

    # ---------------------------------------------------------------------
    # Saving / Loading helpers (reuse HF weights)
    # ---------------------------------------------------------------------
    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    
    def get_compiler_args(self) -> str:
        return "--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor=2 --vectorize-strided-dma' -O1"

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, cfg: InferenceConfig) -> dict:
        return convert_hf_to_neuron_state_dict(state_dict, cfg)
    
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.text_config.image_token_id
        video_token_id = self.config.text_config.video_token_id
        vision_start_token_id = self.config.text_config.vision_start_token_id
        #mrope_position_deltas = []
        mrope_position_deltas_list: List[torch.Tensor] = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0], #zz update
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                            #image_grid_thw[0][0],
                            #image_grid_thw[0][1],
                            #image_grid_thw[0][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(dtype=position_ids.dtype, device=position_ids.device)
                #position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                #    position_ids.device
                #)
                #mrope_position_deltas.append(
                mrope_position_deltas_list.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas_list, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    
    def get_vision_attention(self, seq_length, image_grid_thw):
        
        cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        
        
        # attention_mask = torch.full(
        #         [1, seq_length, seq_length], torch.finfo(torch.float32).min,  dtype=torch.float32
        #     )
        
            
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        
        #Note new vision attention module expects bools
        attention_mask = torch.full(
                [1, seq_length, seq_length], 0,  dtype=torch.float32
            )
        
            
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 1
        
        attention_mask = attention_mask.to(torch.bool)
        return attention_mask
    
    def rot_pos_emb(self, seq_length, grid_thw, head_dim):
        pos_ids = []
        
        rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        spatial_merge_size = 2
        
        # output_rot = torch.zeros((1296, 11))
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        
        output_rot = torch.ones((seq_length, rotary_pos_emb.shape[1]), dtype=rotary_pos_emb.dtype)
        output_rot[:rotary_pos_emb.shape[0],:] = rotary_pos_emb

        return output_rot
    
    def get_vision_cos_sin(self, seq_length, image_grid_thw, head_dim ):
        pos_embeddings = self.rot_pos_emb(seq_length, image_grid_thw, head_dim)
        emb = torch.cat((pos_embeddings, pos_embeddings), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        seq_ids: Optional[torch.LongTensor] = None,
        sampling_params: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        adapter_ids: Optional[torch.FloatTensor] = None,
        medusa_args=None,
        return_dict: Optional[bool] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        aspect_ratios: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        num_chunks: Optional[torch.Tensor] = None,
        has_image: Optional[torch.Tensor] = None,
        vision_key_values: Optional[List[torch.FloatTensor]] = None,
        llava_args: Optional[List] = [],
        input_capture_hook: Optional[Callable] = None,
        prev_hidden: Optional[torch.Tensor] =None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        logging.info(f"IN CAUSUAL LM FORWARD")
        logging.info(f"{input_ids}")

        if self.async_mode:
            # derive future cpu inputs from current cpu inputs
            # if position_ids.shape[1] == input_ids.shape[1]:
            #     next_position_ids = torch.amax(position_ids, 1, keepdim=True)
            # else:
            #     next_position_ids = position_ids

            # next_position_ids = next_position_ids + 1
            # next_attention_mask = self._infer_attention_mask(next_position_ids)
            # self.next_cpu_inputs = {
            #     "attention_mask": next_attention_mask,
            #     "position_ids": next_position_ids,
            # }
            raise Exception("Async mode is not supported")

        sampling_params = (
            self.default_sampling_params if sampling_params is None else sampling_params
        )
        self.sampling_params = sampling_params

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        # infer attention_mask from position_ids if not provided
        # if attention_mask is None:
        #     attention_mask = self._infer_attention_mask(position_ids)

        # self._log_input(input_ids, attention_mask, position_ids, seq_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        if self.async_mode:
            # bs, _ = input_ids.shape
            # outputs, is_run_on_neuron = self._get_model_outputs_async(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     seq_ids=seq_ids,
            #     sampling_params=sampling_params,
            #     adapter_ids=adapter_ids,
            #     medusa_args=medusa_args,
            #     llava_args=llava_args,
            #     pixel_values=(
            #         pixel_values if input_ids.shape[-1] > 1 else torch.tensor([0] * bs)
            #     ),
            #     aspect_ratios=aspect_ratios,
            #     vision_mask=vision_mask,
            #     num_chunks=num_chunks,
            #     has_image=has_image,
            #     prev_hidden=None,
            # )
            raise Exception("Async mode is not supported")
        else:
            outputs, is_run_on_neuron = self._get_model_outputs(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
                pixel_values,
                # pixel_values_videos,
                image_grid_thw,
                #self.rope_deltas
                # video_grid_thw,
                # aspect_ratios,
                # vision_mask,
                # num_chunks,
                # has_image,
            )
            #self.rope_deltas = rope_deltas

        generation_model = self.get_generation_model()
        if not generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        if is_run_on_neuron:
            # When run on neuron, KV cache remains on device
            logits_or_next_tokens = outputs
        else:
            # When run on cpu, KV cache is returned which has to be ignored
            logits_or_next_tokens, *_ = outputs

        return self._construct_output(logits_or_next_tokens)

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        #rope_deltas: Optional[torch.Tensor] = None,
        # video_grid_thw: Optional[torch.LongTensor] = None,
        # aspect_ratios: Optional[torch.LongTensor] = None,
        # vision_mask:,
        # num_chunks,
        # has_image,
    ):
        bs, _ = input_ids.shape
        
        
       
        
        if input_ids.shape[-1] > 1:  # context encoding
            # First time we run, we should calculate correct position id based on content type
            #position_ids = None
            
            seq_length = self.config.neuron_config.max_context_length
            max_image_pixels = PIXEL_SIZE_MAP[seq_length]
            
            self.vision_attention = self.get_vision_attention(max_image_pixels, image_grid_thw)
            self.vision_cos, self.vision_sin = self.get_vision_cos_sin(max_image_pixels, image_grid_thw, self.config.vision_config.embed_dim // self.config.vision_config.num_heads)
            
            rotary_position_id, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, None, 
                attention_mask
                
            )   
            
            max_image_tokens = max_image_pixels//4
            
            
            positions, valid_mask = prepare_scatter_positions(input_ids, self.config.text_config.image_token_id, max_image_tokens)         
            
            self.positions = positions
            self.valid_mask = valid_mask
            
            
            
            
            #Take rope delta by taking into account the current padding            
            current_location = attention_mask.sum(dim=-1)
            # current_location = position_ids.max(dim=-1).values
            #One rope delta for each item in batch
            max_position_id = rotary_position_id.max(dim=2).values
            rope_deltas = max_position_id.max(dim=0, keepdim=True).values - current_location + 1
            
            self.rope_deltas = rope_deltas
            
            self.max_rot_position = max_position_id
            
            
                        
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params[seq_ids],
                pixel_values,
                image_grid_thw,
                rotary_position_id,
                self.vision_attention,
                self.vision_cos,
                self.vision_sin,
                self.positions,
                self.valid_mask
               
            )
            
            
            # rope_deltas_out = outputs[-1]
            # outputs = outputs[:-1]
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()

        else:  # token generation
            if (
                self.next_cpu_inputs is not None and self.prior_outputs is not None
            ):  # this is never not None and not in async mode
                _input_ids = self.prior_outputs
                _attention_mask = self.next_cpu_inputs["attention_mask"]
                _position_ids = self.next_cpu_inputs["position_ids"]
            else:
                _input_ids = input_ids
                _attention_mask = attention_mask
                #Update position ids
                # _position_ids = position_ids + self.rope_deltas
                
                
            
                
           
            # rotary_position_id = position_ids.add(self.rope_deltas)
            # rotary_position_id = rotary_position_id.unsqueeze(0).expand(3, -1, -1)
            rotary_position_id = self.max_rot_position.add(1)
            # print(f"self.max_rot_position, {self.max_rot_position.shape}")
            
            rotary_position_id = rotary_position_id.view(3,1,1)
            self.max_rot_position = self.max_rot_position.add(1)
            
            # rotary_position_id = (self.rope_pos + 1).unsqueeze(0).expand(3,1,1)
            # self.rope_pos = self.rope_pos + 1
            # print("rotary_pos_ids", rotary_position_id)
            
            
            #zz update   
            #image_grid_thw=image_grid_thw[[0]]

            outputs = self.token_generation_model(
                _input_ids,
                _attention_mask,
                position_ids,
                seq_ids,
                sampling_params[seq_ids],
                torch.ones((1,1), dtype=torch.int32),
                image_grid_thw,
                rotary_position_id,
                torch.ones((1,1,1), dtype=torch.bool),#self.vision_attention,
                torch.ones((1,1), dtype=torch.int32),#self.vision_cos,
                torch.ones((1,1), dtype=torch.int32),#self.vision_sin,
                torch.ones((1,1), dtype=torch.int32),
                torch.ones((1,1), dtype=torch.bool)
                
            )
            
            
            
            # print("TOKEN GENERATION COMPLETE")
            is_run_on_neuron = self.token_generation_model.is_neuron()
        return outputs, is_run_on_neuron

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        #These weights are shared
        state_dict["text_model.lm_head.weight"] = state_dict["text_model.embed_tokens.weight"].clone()
        
    
    def get_required_kwargs(self) -> List[str]:
        """The list of additional input arguments to be prepared in HuggingFaceGenerationAdapter.prepare_inputs_for_generation()"""
        return [
            "image_grid_thw",
            "pixel_values"
          
        ]

    # All other behaviour (async gen, sampler, kv‑cache) inherited unchanged
