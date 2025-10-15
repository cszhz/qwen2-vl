import torch
from torch import nn
from torch import Tensor
from torch.nn import LayerNorm
from typing import Optional, Tuple
from transformers.activations import ACT2FN

from neuronx_distributed_inference.modules.attention.utils import _rotate_half
from neuronx_distributed_inference.models.config import (  # noqa: E402
    InferenceConfig,
    to_dict,
)
from neuronx_distributed_inference.modules.attention.gqa import (
    GroupQueryAttention_QKV,
    GroupQueryAttention_O,
)
import os
import math
import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed.parallel_layers import parallel_state, utils

from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed_inference.modules.attention.utils import move_heads_front, repeat_kv
import logging

logger = logging.getLogger(__name__)



class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionAttention(NeuronAttentionBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.embed_dim
        self.num_attention_heads = config.num_heads
        self.num_key_value_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tp_degree = config.neuron_config.tp_degree
        # self.tp_degree = 1
        self.torch_dtype = config.neuron_config.torch_dtype
        self.fused_qkv = True
        self.clip_qkv = None
        self.bias = True
        self.padding_side = "right"
        self.o_proj_layer_name = "proj"
        
        
        self.init_gqa_properties()
        
    def init_gqa_properties(self):
        self.qkv = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            logical_nc_config=self.neuron_config.logical_nc_config,
            qkv_kernel_nbsd_layout=self.neuron_config.qkv_kernel_nbsd_layout,
            on_cpu=self.neuron_config.on_cpu,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
        )
        self.num_heads = utils.divide(self.qkv.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(
            self.qkv.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.logical_nc_config = self.neuron_config.logical_nc_config

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value = None,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """take care of the shape, layout, group query, custom position encoding, etc."""
        Q, K, V = self.qkv(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids
        )
        

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz = 1
        if hidden_states.dim() == 2:
            q_len, _ = hidden_states.size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        
        
        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        #print(f"IN PREP QKV: Q SIZE: {Q.shape}")
        #print(f"IN PREP QKV: K SIZE: {K.shape}")
        
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        cos, sin = position_ids
        
        Q, K = apply_rotary_pos_emb_vision(Q.transpose(1,2), K.transpose(1,2), cos, sin)

        return Q.transpose(1,2), K.transpose(1,2), V
    
    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")
        
        #I have disabled flash attention for vision encoder
        #This is because flash attn kernel always uses attention mask all 1s which is not true for our padded images
        #TODO: find better way to handle padded images

        # if flash_attn_strategy != FlashAttentionStrategy.NONE:
        #     logger.debug(f"ATTN kernel: logical_nc_config={self.logical_nc_config}")
        #     # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
        #     # because flash attention does not use attention_mask). In practice, we use right
        #     # padding so this is unlikely to cause issues
        #     assert self.padding_side == "right" or bsz == 1

        #     # original shape of q, k, v is BHSD, and expected output is also BHSD.
        #     logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
        #     # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
        #     # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.

        #     # original Q shape: batch, num_heads, seqlen, d_head
        #     Q = (
        #         Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
        #         .reshape((bsz * self.num_heads, self.head_dim, q_len))
        #         .to(self.torch_dtype)
        #     )
        #     Q = Q / math.sqrt(self.head_dim)
        #     K_active = (
        #         K_active.permute(0, 1, 3, 2)
        #         .reshape((bsz * self.num_heads, self.head_dim, q_len))
        #         .to(self.torch_dtype)
        #     )
        #     V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
        #         self.torch_dtype
        #     )
        #     # shape: (B*H)DS
        #     attn_output = torch.zeros(
        #         bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
        #     )

        #     logger.debug("Input parameter shapes")
        #     logger.debug(f"Q input shape {Q.shape}")
        #     logger.debug(f"K input shape {K_active.shape}")
        #     logger.debug(f"V input shape {V_active.shape}")
        #     logger.debug(f"Attn output shape {attn_output.shape}")

        #     if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
        #         grid = (nc(self.logical_nc_config),)

        #         _flash_fwd_call[grid](
        #             Q,
        #             K_active,
        #             V_active,
        #             1.0,
        #             attn_output,
        #             kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
        #         )
        #     elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
        #         _flash_fwd_call(
        #             Q,
        #             K_active,
        #             V_active,
        #             1.0,
        #             attn_output,
        #             kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
        #         )
        #     else:
        #         raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

        #     # shape: BHDS
        #     attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
        #     logger.debug(f"Attn output after reshape {attn_output.shape}")
        # else:
        logger.debug("ATTN: native compiler")
        logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
        active_scores = self.scaled_qk(Q, K_active, attention_mask)
        active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
            Q.dtype
        )
        attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Implements each layer's forward pass for the attention block."""
        bsz = 1
        if hidden_states.dim() ==2:
            q_len, _ = hidden_states.size()
            hidden_states = hidden_states.unsqueeze(0)
        else:
            bsz, q_len, _ = hidden_states.size()
            
        
        Q, K, V = self.prep_qkv_tensors(
            position_embeddings,
            hidden_states,
            None,
            adapter_ids=None,
            cos_cache=None,
            sin_cache=None,
            rmsnorm=None,
        )

        attn_output, flash_attn_strategy = self.perform_prefill(
            Q, K, V, q_len, bsz, attention_mask
        )


        # transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output, adapter_ids=None)

        return attn_output[0,:,:]



class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, vision_config) -> None:
        super().__init__()
        self.norm1 = LayerNorm(vision_config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(vision_config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(vision_config.embed_dim * vision_config.mlp_ratio)
        
        self.attn = VisionAttention(vision_config)
        
        self.mlp = VisionMlp(
            dim=vision_config.embed_dim, 
            hidden_dim=mlp_hidden_dim, 
            hidden_act=vision_config.hidden_act)

   
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NeuronQwen2VisionTransformerPretrainedModel(nn.Module):
    #config_class = Qwen2VLVisionConfig
    #_no_split_modules = ["Qwen2VLVisionBlock"]

    def __init__(self, vision_config: InferenceConfig ) -> None:
        super().__init__()
        self.spatial_merge_size = vision_config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.embed_dim,
        )

        head_dim = vision_config.embed_dim // vision_config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(vision_config) for _ in range(vision_config.depth)]
        )
        self.merger = PatchMerger(
            dim=vision_config.hidden_size, 
            context_dim=vision_config.embed_dim,
            spatial_merge_size=vision_config.spatial_merge_size
        )

    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        
       
        return rotary_pos_emb
       
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        
        
        hidden_states = self.patch_embed(hidden_states)
        
        position_embeddings = (cos, sin)
        
        for blk in self.blocks:
            hidden_states = blk(hidden_states, attention_mask, position_embeddings=position_embeddings)
        
        
        return self.merger(hidden_states)


