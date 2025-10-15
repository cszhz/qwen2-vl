# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Set

import torch
from neuronx_distributed_inference.models.mllama.image_transform import (
    custom_image_preprocessing)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params)
from neuronx_distributed_inference.modules.lora_serving import (
    LoraCheckpoint, LoraServingConfig)
from PIL.Image import Image as PILImage

from vllm.config import VllmConfig
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuronx_distributed import (
    _get_model_architecture, get_neuron_model)
from vllm.multimodal import MultiModalKwargs
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.worker.neuron_model_runner import (ModelInputForNeuron,
                                             NeuronModelRunner)

logger = init_logger(__name__)


class NeuronxDistributedModelRunner(NeuronModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        super().__init__(vllm_config)
        self.lora_checkpoint = None
        self.model = None
        self.lora_serving_config = None

    @staticmethod
    def _get_lora_paths_strings(lora_modules: List[LoRAModulePath]):
        if not lora_modules:
            return None
        return {_.name: _.path for _ in lora_modules}

    def _get_nxdi_lora_config(self):
        override_neuron_config = self.model_config.override_neuron_config
        lora_modules = override_neuron_config.pop("lora_modules", None)
        target_modules = override_neuron_config.pop("target_modules", None)
        lora_ckpt_paths = self._get_lora_paths_strings(lora_modules)
        if self.lora_config.max_loras < len(lora_ckpt_paths):
            raise ValueError(
                "Number of LoRAs (%s) exceeds maximum "
                "allowed (%s)", len(lora_ckpt_paths),
                self.lora_config.max_loras)

        return LoraServingConfig(
            max_loras=self.lora_config.max_loras,
            max_lora_rank=self.lora_config.max_lora_rank,
            target_modules=target_modules,
            lora_ckpt_paths=lora_ckpt_paths,
        )

    def load_model(self) -> None:
        # Update LoRA config
        if self.lora_config is not None:
            self.lora_serving_config = self._get_nxdi_lora_config()
            self.lora_checkpoint = LoraCheckpoint(self.lora_serving_config)
        self.model = get_neuron_model(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            lora_serving_config=self.lora_serving_config)

    def get_nxd_sampling_params(self, sampling_metadata):
        if self.model.config.neuron_config.on_device_sampling_config:
            max_topk = (self.model.config.neuron_config.
                        on_device_sampling_config.global_topk)
        else:
            max_topk = self.model.config.vocab_size

        top_k = [1] * self.scheduler_config.max_num_seqs
        top_p = [1.0] * self.scheduler_config.max_num_seqs
        temperature = [1.0] * self.scheduler_config.max_num_seqs

        for index, sequenceGroupToSample in enumerate(
                sampling_metadata.seq_groups):
            top_k[index] = (sequenceGroupToSample.sampling_params.top_k
                            if sequenceGroupToSample.sampling_params.top_k > 0
                            else max_topk)
            top_p[index] = sequenceGroupToSample.sampling_params.top_p
            temperature[index] = (
                sequenceGroupToSample.sampling_params.temperature)

        sampling_params = prepare_sampling_params(
            batch_size=self.scheduler_config.max_num_seqs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature)
        return sampling_params

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "NeuronModelRunner does not support multi-step execution.")

        if _get_model_architecture(
                self.model.config) != "MllamaForConditionalGeneration":
            return super().execute_model(model_input, kv_caches,
                                         intermediate_tensors, num_steps)

        sampling_params = self.get_nxd_sampling_params(
            model_input.sampling_metadata)

        if model_input.multi_modal_kwargs.get('image') is not None:
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                seq_ids=model_input.input_block_ids,
                pixel_values=model_input.multi_modal_kwargs.get(
                    'pixel_values'),
                aspect_ratios=model_input.multi_modal_kwargs.get(
                    'aspect_ratios'),
                sampling_params=sampling_params,
                num_chunks=model_input.multi_modal_kwargs.get('num_chunks'),
                has_image=model_input.multi_modal_kwargs.get('has_image'),
            )
        else:
            empty_pixel_values = torch.zeros([1, 1, 4, 3, 560, 560],
                                             dtype=torch.bfloat16)
            empty_aspect_ratios = torch.ones([1, 1, 2], dtype=torch.int64)
            num_chunks = torch.tensor([[1]
                                       ])  # dummy num_chunks, will not be used
            has_image = torch.tensor([0])
            hidden_states = self.model(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                seq_ids=model_input.input_block_ids,
                pixel_values=empty_pixel_values,
                aspect_ratios=empty_aspect_ratios,
                sampling_params=sampling_params,
                num_chunks=num_chunks,
                has_image=has_image,
            )

        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]

    def process_multi_modal_data_neuron(self, mm_data):
        pixel_values_list = []
        aspect_ratios_list = []
        num_chunks_list = []
        has_image_list = []
        images = mm_data.get('image')

        if isinstance(images, PILImage):
            images = [images]
        if isinstance(images, torch.Tensor):
            images = [images]

        if images is not None:
            for input_image in images:
                if isinstance(input_image, PILImage):
                    pixel_values, aspect_ratios, num_chunks = \
                        custom_image_preprocessing(self.model.config, \
                                                   [[input_image]])
                    has_image = torch.tensor([1])
                    image_tensors = [pixel_values.bfloat16().clone().detach(), \
                                     aspect_ratios, num_chunks, has_image]
                else:
                    empty_pixel_values = torch.zeros([1, 1, 4, 3, 560, 560], \
                                                     dtype=torch.bfloat16)
                    empty_aspect_ratios = torch.ones([1, 1, 2], dtype=torch.int64)
                    # dummy num_chunks, will not be used
                    num_chunks = torch.tensor([[1]])
                    has_image = torch.tensor([0])
                    image_tensors = [empty_pixel_values, empty_aspect_ratios, \
                                     num_chunks, has_image ]

                pixel_values_list.append(image_tensors[0])
                aspect_ratios_list.append(image_tensors[1])
                num_chunks_list.append(image_tensors[2])
                has_image_list.append(image_tensors[3])

            mm_data["pixel_values"] = torch.cat(pixel_values_list,
                                                dim=0).squeeze(0)
            mm_data["aspect_ratios"] = torch.cat(aspect_ratios_list, dim=0)\
                .squeeze(0)
            mm_data["num_chunks"] = torch.cat(num_chunks_list, dim=0).squeeze(0)
            mm_data["has_image"] = torch.cat(has_image_list, dim=0).squeeze(0)
        else:
            print(f"Image key not in mm_data; continuing with input as-is")

        return mm_data

    def _get_lora_adapter_ids(self, seq_group_metadata_list):
        # set LoRA adapter IDs for multi-lora serving
        batch_size = len(seq_group_metadata_list)
        if self.lora_checkpoint is not None:
            # "0" indicates NxDI to use the base model for inference
            adapter_ids = ["0"] * batch_size
            for idx, seq_group_metadata in enumerate(seq_group_metadata_list):
                if seq_group_metadata.lora_request is not None:
                    adapter_ids[
                        idx] = seq_group_metadata.lora_request.lora_name

            # convert adapter_ids from strings to integers
            adapter_ids = self.lora_checkpoint.convert_adapter_ids_to_indices(
                adapter_ids, batch_size)
        else:
            adapter_ids = torch.zeros((batch_size), dtype=torch.int32)

        return adapter_ids

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_block_ids, seq_lens,
             multi_modal_kwargs
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_block_ids) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None

        if not self._on_device_sampling_disabled:
            for seq_group_metadata in seq_group_metadata_list:
                sampling_params = seq_group_metadata.sampling_params
                top_k, top_p, temperature = (
                    self._convert_to_neuron_sampling_params(sampling_params))
                sampling_params.top_k = top_k
                sampling_params.top_p = top_p
                sampling_params.temperature = temperature

        # we need multi_modal_data for later tokens as well
        multi_modal_inputs_list: List[MultiModalKwargs] = []
        for seq_group_metadata in seq_group_metadata_list:
            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                multi_modal_inputs_list.append(mm_data)
        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_inputs_list)

        lora_adapter_ids = self._get_lora_adapter_ids(seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since neuron worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))

        return ModelInputForNeuron(input_tokens=input_tokens,
                                   input_positions=input_positions,
                                   input_block_ids=input_block_ids,
                                   sampling_metadata=sampling_metadata,
                                   multi_modal_kwargs=multi_modal_kwargs,
                                   adapter_ids=lora_adapter_ids)

    def remove_all_loras(self):
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def add_lora(self, lora_request: LoRARequest):
        logger.warning(
            "Adding LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server. If you supplied "
            "the parameter, you can ignore this warning. Ignoring"
            "lora request: ", lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(
            "Managing LoRAs is only supported through the "
            "lora_modules parameter to the LLM Class"
            " or --lora-modules while using online server)")
