import warnings

import torch
import torch.nn.functional as F

from neuronx_distributed_inference.models.model_wrapper import (
    DecoderModelInstance,
    ModelWrapper,
    CONTEXT_ENCODING_MODEL_TAG
)
from neuronx_distributed_inference.models.qwen2_vl.utils import (
    PIXEL_SIZE_MAP, update_messages, pad_inputs
)
from neuronx_distributed_inference.modules.generation.sampling import (
    prepare_sampling_params,
)

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from model_config import messages


class ModelWrapperQwen2VL(ModelWrapper):

    def input_generator(self):
        model_path = "Qwen/Qwen2-VL-7B-Instruct"

        inputs = []
        for bucket in self.neuron_config.buckets:
            n_active_tokens = (
                bucket
                if self.neuron_config.bucket_n_active_tokens
                else self.neuron_config.n_active_tokens
            )
            max_image_pixels = PIXEL_SIZE_MAP[bucket]
            max_image_tokens = max_image_pixels//4
            img_start_token_id = 151652
            image_token_id = 151655
            img_end_token_id = 151653
            pad_token_id = 151643
            patch_size = 14
            global messages
            messages = update_messages(messages, PIXEL_SIZE_MAP, bucket, patch_size)
            #print("warm up:",messages)
            # Preparation for inference

            processor = AutoProcessor.from_pretrained(model_path)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs_hf = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # input_ids = torch.ones((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            if n_active_tokens > 1:
                input_ids = inputs_hf["input_ids"]
                # input_ids, _ = pad_inputs(input_ids, torch.ones((1, input_ids.shape[1])), img_start_token_id, img_end_token_id, image_token_id, max_image_tokens)
                
                n_pad = bucket - input_ids.shape[1]
                input_ids = torch.nn.functional.pad(input_ids, (0, n_pad), "constant", pad_token_id)
                
                
                n_vision = torch.sum(input_ids[0,:]==image_token_id)
                # assert n_vision == max_image_tokens
                positions = torch.ones(1, max_image_pixels//4, dtype=torch.int32)
                valid_mask = torch.ones(1, max_image_pixels//4, dtype=torch.bool)
                
                pixel_values = inputs_hf["pixel_values"]
            
                correct_shape_pixel_values = torch.zeros((max_image_pixels,1176))

                correct_shape_pixel_values[:pixel_values.shape[0],:] = pixel_values
                vision_attention = torch.ones((1,max_image_pixels,max_image_pixels))
                vision_attention = vision_attention.to(torch.bool)
                vision_cos = torch.ones((max_image_pixels, 80))
                vision_sin = torch.ones((max_image_pixels, 80))
                
                

                
            else:
                input_ids = torch.ones((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
                positions = torch.ones(1, 1, dtype=torch.int32)
                valid_mask = torch.ones(1, 1, dtype=torch.bool)
                correct_shape_pixel_values = torch.zeros((1,1), dtype=torch.int32)
                vision_attention = torch.ones((1,1,1), dtype=torch.int32)
                vision_attention = vision_attention.to(torch.bool)
                vision_cos = torch.ones((1, 1),dtype=torch.int32)
                vision_sin = torch.ones((1, 1),dtype=torch.int32)
                

                
            input_ids = input_ids.to(torch.int32)

            
            
            image_grid_thw = inputs_hf["image_grid_thw"]

            attention_mask = torch.ones(
                    (self.neuron_config.batch_size, bucket), dtype=torch.int32
                )
            

        
            position_ids = torch.arange(n_active_tokens, dtype=torch.int32)
            position_ids = position_ids.expand(self.neuron_config.batch_size, n_active_tokens)

            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)

            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros(
                (self.neuron_config.batch_size, sampling_params_len),
                dtype=torch.float32,
            )


            rotary_position_ids = position_ids.expand(3, self.neuron_config.batch_size, n_active_tokens)
            input_data = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
            )
            
            
            
            
            
           
            if not self.neuron_config.skip_vision:
                input_data += (correct_shape_pixel_values, image_grid_thw)
            
            
            input_data += (rotary_position_ids, vision_attention, vision_cos, vision_sin, positions, valid_mask)

            inputs.append(input_data)
            
            return inputs
       
    def pad_inputs(self, *args, pad_type="first_fit"):
        """
        The following padding strategies are supported:

        1) "max": Pad to the max bucket length
            ex. If we have input_length = 8 and buckets [5, 15, 25, 35] we choose 35
            because 35 is the last (highest) bucket.
        2) "first_fit" (default): Pad to the nearest highest bucket.
            ex. If we have input_length = 8 and buckets [5, 15, 25, 35] we choose 15
            because while 8 is closer to bucket 5, we pad up, so the nearest highest is 15
        3) "second_fit": Pad to the second nearest highest bucket.
            ex. If we have input_length = 8 and buckets [5, 15, 25, 35] we choose 25
            because 15 is the next bucket, and 25 follows 15 as the "next next"
            or "second fit" bucket.
        """
        VALID_PAD_TYPES = {"max", "first_fit", "second_fit"}
        assert (
            pad_type in VALID_PAD_TYPES
        ), f"Found {pad_type=}, when it should be one of: {VALID_PAD_TYPES=}"
        pad_length = (
            self.neuron_config.max_context_length
            if self.tag == CONTEXT_ENCODING_MODEL_TAG
            else self.neuron_config.max_length
        )

        if pad_type == "first_fit" or pad_type == "second_fit":
            pad_length = self.get_target_bucket(*args, strategy=pad_type)

        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            to_pad = args[:3]
            pad_lengths = [pad_length - arg.shape[1] for i, arg in enumerate(to_pad)]
            tensor_pad_vals = [self.config.pad_token_id, 0, 1]
            
            if any([pad_len < 0 for pad_len in pad_lengths]):
                if self.neuron_config.allow_input_truncation:
                    warnings.warn(
                        f"Truncating input ({to_pad[1].shape[1]} tokens) to max_context_length ({self.neuron_config.max_context_length} tokens). This may cause unexpected outputs."
                    )
                else:
                    raise ValueError(
                        f"Inputs supplied ({to_pad[1].shape[1]} tokens) are longer than max_context_length ({self.neuron_config.max_context_length} tokens). To truncate inputs, set allow_input_truncation=True."
                    )

            padded_args = [
                F.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)
            ]
            args = (*padded_args, *args[3:])

            #Padding for rotary embeddings
            pad_length_rot = pad_length - args[7].shape[2]
            padded_args = F.pad(args[7], (0, pad_length_rot), "constant", 1)
            args = (*args[:7], padded_args, *args[8:])
            
            
            
            if self.is_chunked_prefill:
                args = list(args)

                # Pad arguments and stack List args.
                cp_max_num_seqs = self.neuron_config.cp_max_num_seqs
                max_context_length = self.neuron_config.max_context_length
                max_total_len = (
                    self.neuron_config.cp_num_active_blocks * self.neuron_config.pa_block_size
                    + self.neuron_config.max_context_length
                )

                cache_to_ctx = args[-1]
                cache_to_ctx = F.pad(cache_to_ctx, [0, max_total_len - cache_to_ctx.shape[0]])
                args[-1] = cache_to_ctx

                active_to_ctx = args[-2]
                active_to_ctx = F.pad(active_to_ctx, [0, max_total_len - active_to_ctx.shape[0]])
                args[-2] = active_to_ctx

                cache_mask = args[-3]
                cache_mask = F.pad(cache_mask, [0, max_total_len - cache_mask.shape[0]])
                args[-3] = cache_mask

                context_lens = args[-4]
                context_lens = F.pad(context_lens, [0, cp_max_num_seqs - context_lens.shape[0]])
                args[-4] = context_lens

                num_queries = args[-5]
                num_queries = F.pad(num_queries, [0, cp_max_num_seqs - num_queries.shape[0]])
                args[-5] = num_queries

                active_block_table = args[-6]
                cp_num_active_blocks = self.neuron_config.cp_num_active_blocks
                active_block_table = F.pad(
                    active_block_table, [0, cp_num_active_blocks - len(active_block_table)]
                )
                args[-6] = active_block_table

                slot_mapping = args[-7]
                # need to be padded by -1 to avoid overriding existing KV cache.
                slot_mapping = F.pad(
                    slot_mapping, [0, max_context_length - slot_mapping.shape[0]], value=-1
                )
                args[-7] = slot_mapping
                args = tuple(args)
        else:
            input_ids, attention_mask, *rest_of_args = args
            pad_len = pad_length - attention_mask.shape[1]

            if pad_len < 0:
                if self.neuron_config.allow_input_truncation:
                    warnings.warn(
                        f"Truncating attention mask (length={attention_mask.shape[1]}) to max_length ({self.neuron_config.max_length} tokens). This may cause unexpected outputs."
                    )
                else:
                    raise ValueError(
                        f"Attention mask supplied (length={attention_mask.shape[1]}) is longer than max_length ({self.neuron_config.max_length} tokens). To truncate attention mask, set allow_input_truncation=True."
                    )

            padded_attention_mask = F.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids, padded_attention_mask, *rest_of_args)
            
            
       
        
        #
       
        return args
    

    def get_model_instance(self):
        return Qwen2VLDecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            **self.model_init_kwargs,
        )


class Qwen2VLDecoderModelInstance(DecoderModelInstance):
    pass
