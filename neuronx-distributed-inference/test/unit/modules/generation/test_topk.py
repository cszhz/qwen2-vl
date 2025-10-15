import unittest
from functools import partial

import pytest
import torch
from neuronx_distributed.operators.topk import topk as nxd_topk
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, SPMDRank
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch import nn
from torch_neuronx.utils import get_platform_target

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.utils.distributed import get_tp_group


class EmbeddingModel(nn.Module):
    def __init__(self, is_distributed=False, config=None):
        super().__init__()
        self.config = config
        if is_distributed:
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
                sequence_dimension=2,
            )
            self.embed_tokens.rank_util = SPMDRank(world_size=config.neuron_config.tp_degree)

        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.config.pad_token_id,
                dtype=config.neuron_config.torch_dtype,
            )
        self.embed_tokens.training = False

    @torch.inference_mode()
    def forward(self, input_ids):
        logits = self.embed_tokens(input_ids)
        values, indices = nxd_topk(
            logits,
            k=256,
            gather_dim=2,
            dim=2,
            stages=self.config.stages,
            rank_id=self.embed_tokens.rank_util.get_rank(),
        )
        return values, indices.to(torch.int32)

    def cpu_forward(self, input_ids):
        logits = self.embed_tokens(input_ids)
        values, indices = torch.topk(logits, k=256, dim=2)
        return values, indices.to(torch.int32)


@pytest.mark.xfail(reason="not valid test for trn1.2xlarge", strict=False)
@pytest.mark.parametrize(
    "batch_size, sequence_length, tp_degree, dtype, stages",
    [
        (1, 1, 64, torch.float32, 3),
        (1, 1, 32, torch.float32, 2),
        (1, 1, 32, torch.float32, 1),
        (1, 1, 64, torch.float32, 1),
    ],
)
def test_vocab_parallel_embedding(batch_size, sequence_length, tp_degree, dtype, stages):
    hardware = get_platform_target()
    if hardware == "trn1" and tp_degree == 64:
        pytest.skip("Not supported in trn1")
    if hardware == "trn2" and tp_degree == 32:
        pytest.skip("Not supported in trn2")

    def get_ckpt():
        model_sd = torch.load("/tmp/model.pt")
        model_sd["embed_tokens.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return model_sd

    input_shape = (batch_size, sequence_length)
    config_dict = {
        "hidden_size": 128256,
        "num_attention_heads": 32,
        "num_hidden_layers": 1,
        "num_key_value_heads": 8,
        "pad_token_id": 0,
        "vocab_size": tp_degree,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-05,
        "hidden_act": "silu",
        "tp_degree": tp_degree,
        "torch_dtype": dtype,
        "batch_size": batch_size,
        "vocab_parallel": True,
        "sequence_parallel_enabled": True,
        "stages": stages,
    }
    neuron_config = NeuronConfig(**config_dict)
    config = InferenceConfig(neuron_config=neuron_config, **config_dict)
    print(f"stages: {config.stages}")
    model = partial(EmbeddingModel, is_distributed=False, config=config)()
    torch.save(model.state_dict(), "/tmp/model.pt")
    model.load_state_dict(model.state_dict())

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=get_ckpt,
        num_cores_per_group=1,
        logical_nc_config=2,
    )
    model_cls = partial(EmbeddingModel, is_distributed=True, config=config)

    input_ids = torch.randint(0, config.vocab_size, input_shape, dtype=torch.int32)

    builder.add(
        key="main",
        model_instance=BaseModelInstance(module_cls=model_cls, input_output_aliases={}),
        example_inputs=[(input_ids,)],
        priority_model_idx=0,
    )

    traced_model = builder.trace(initialize_model_weights=True)
    traced_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
    traced_output = traced_model(input_ids)

    model.eval()
    cpu_output = model.cpu_forward(input_ids)
    torch.testing.assert_close(
        cpu_output,
        tuple([x.to("cpu") for x in traced_output]),
        atol=1e-3,
        rtol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
