import os
import torch


from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
from qwen_vl_utils import process_vision_info
from neuronx_distributed_inference.models.qwen2_vl.modelling_qwen2_vl import (
   Qwen2VLNeuronConfig,
   Qwen2VLInferenceConfig,
   NeuronQwen2VLForCausalLM,
   
)
from neuronx_distributed_inference.models.qwen2_vl.utils import (
   PIXEL_SIZE_MAP, update_messages, pad_inputs
)
from neuronx_distributed_inference.utils.hf_adapter import (
   load_pretrained_config,HuggingFaceGenerationAdapter
   
)

from model_config import messages,num_img_per_prompt



def run_qwen2_generate():
    model_path='/home/ubuntu/qwen2-vl/neuronx-distributed-inference/examples/qwen2_vl/models/Qwen/Qwen2-VL-7B-Instruct'
    traced_model_path='./traced_models'
    torch.manual_seed(0)

    processor = AutoProcessor.from_pretrained(model_path)

    global messages
    tp_degree=2
    batch_size = 1
    #seq_len = 4096 #1 image
    seq_len = 8192 #2 image
    #seq_len = 32768 #2 image
    #seq_len = 32768 #2 image
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.max_new_tokens = 200
    #generation_config.max_tokens = 5000
    generation_config.top_p = 1.0
    generation_config.temperature = 0.2
    generation_config.do_sample = False
    
    
    neuron_config = Qwen2VLNeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        seq_len=seq_len,
        enable_bucketing=False,
        # on_device_sampling=False,
    )

    os.environ["LOCAL_WORLD_SIZE"] = str(int(neuron_config.world_size))
    
    config = Qwen2VLInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )
    config.neuron_config.skip_vision = False


    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.bos_token
    ##if not os.path.exists(traced_model_path):
    # Compile and save model.
    
    traced_model_path=traced_model_path+f'/seq_len_{seq_len}_tp_{tp_degree}-image_{num_img_per_prompt}'

    if True:
        print("\nCompiling and saving model...")
        model = NeuronQwen2VLForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)


    print("\nLoading model from compiled checkpoint...")
    model = NeuronQwen2VLForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)
    tokenizer.pad_token = tokenizer.bos_token


    generation_model = HuggingFaceGenerationAdapter(model)
    
    ##Update the messages to add max pixels supported for this seq_length
    messages = update_messages(messages, PIXEL_SIZE_MAP, seq_len, config.vision_config.patch_size)
    
    # Preparation for inference
    text = processor.apply_chat_template(
       messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
       text=[text],
       images=image_inputs,
       videos=video_inputs,
       padding=True,
       return_tensors="pt",
    )

    
    # Paddding step:
    pixel_values = inputs["pixel_values"]
    #Pad image to fixed shape
    correct_shape_pixel_values = torch.zeros((PIXEL_SIZE_MAP[seq_len],1176))
    correct_shape_pixel_values[:pixel_values.shape[0],:] = pixel_values
    
    max_image_tokens = PIXEL_SIZE_MAP[seq_len]//4 #图像patch编码 - 每个2x2像素块通常编码为1个token
    
    print("DEBUG: 输入形状:", [tensor.shape for tensor in [inputs["input_ids"], inputs["attention_mask"],
    correct_shape_pixel_values, inputs.get('image_grid_thw') if inputs.get('image_grid_thw') is not None else None]])
    print("DEBUG: 输入类型:", [tensor.dtype for tensor in [inputs["input_ids"], inputs["attention_mask"],
    correct_shape_pixel_values, inputs.get('image_grid_thw') if inputs.get('image_grid_thw') is not None else None]])

    generated_ids = generation_model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
        attention_mask = inputs["attention_mask"],
        pixel_values = correct_shape_pixel_values,
        image_grid_thw = inputs.get('image_grid_thw')
       
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print('output_text', output_text)



if __name__ == "__main__":
   run_qwen2_generate()
