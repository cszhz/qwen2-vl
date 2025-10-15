import argparse
import os

from vllm import LLM, SamplingParams, TextPrompt
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from neuronx_distributed_inference.models.qwen2_vl.utils import (
   PIXEL_SIZE_MAP, update_messages, pad_inputs
)

# images = ['/home/ubuntu/qwen2-vl-neuron/examples/image_1024_768.jpg', "/home/ubuntu/qwen2-vl-neuron/examples/dog.jpg"]
# texts = ["What is in this image? Tell me a story", "Describe the image?"]

images = ["/home/ubuntu/qwen2-vl-neuron/examples/dog.jpg"]
texts = ["What is the dividend payout in 2012?"]

os.environ[
    "NEURON_COMPILED_ARTIFACTS"] = "/home/ubuntu/tiktok/examples/traced_models" #"/home/ec2-user/software/grab/qwen2-vl-neuron/examples/traced_models/Qwen2-VL-2B-Instruct"
#os.environ["NEURON_TENSORBOARD_PLUGIN_DUMP_MODEL"] = "1"

def create_llm(args):    
    llm = LLM(
        model=args.model_path,  # MODEL_PATH
        max_num_seqs=args.batch_size,
        max_model_len=args.seq_len,
        block_size=args.seq_len,
        device="neuron",
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=1,
        override_neuron_config={
            "enable_bucketing": False,
        },
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    return llm, processor


def fetch_generation(llm, inputs):

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=512
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated output: {generated_text}")

def add_instruct(prompt: str):
    return f"<|im_start|>system \n You are a helpful assistant.<|im_end|><|im_start|>user \n <|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|><|im_start|>assistant\n"

def create_messages(texts: list, images: list) -> dict:
    messages = []
    for text, image in zip(texts, images):
        message = {}
        message['role'], message['content'] = 'user', []
        message['content'].append({
            "type": "text",
            "text": text
        })
        message['content'].append({
            "type": "image",
            "image": image
        })
        print(f"Appending message: {message}")
        messages.append(message.copy())
    print(f"messages: {messages}")
    return messages

def create_qwen_request_inputs(messages: dict):

    image_inputs, _ = process_vision_info(messages)
    batch_inputs, image_idx = [], 0
    for message in messages:
        text, image = None, None
        for content in message['content']:
            if 'text' in content:
                text = add_instruct(content['text'])
            if 'image' in content:
                image = image_inputs[image_idx]
                image_idx += 1
        assert text != None and image != None
        input = TextPrompt(prompt=text,
                    multi_modal_data={
                        "image": image  # Pass single image tensor
                        }
                    )
        print(f"input: {input}")
        batch_inputs.append(input)

    return batch_inputs

def parse_args():
    parser = argparse.ArgumentParser(description='Model Configuration Parameters')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing (default=1)'
    )
    parser.add_argument(
        '--tensor_parallel_size',
        type=int,
        default=2,
        help='Number of parallel tensors for distributed processing (default=2)'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=128,
        help='Maximum sequence length (default=128)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='/home/ubuntu/Qwen2-VL-7B-Instruct',
        help='Path to the model files (default: /home/ubuntu/Qwen2-VL-7B-Instruct)'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=14,
        help='Patch size of the model images'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    #os.environ["LOCAL_WORLD_SIZE"] = str(args.tensor_parallel_size)

    llm, processor = create_llm(args)

    messages = create_messages(texts, images)
    messages = update_messages(messages, PIXEL_SIZE_MAP, args.seq_len, args.patch_size)             

    batch_inputs = create_qwen_request_inputs(messages)

    fetch_generation(llm, batch_inputs)

