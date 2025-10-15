## Inference Server

### Prerequisites
1. Setup vLLM
```
git clone -b neuron-2.22-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git
cd upstreaming-to-vllm
pip install -r requirements-neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

The changes here are based on vllm commit `049e769ddb5b18d09f4157a258c3d9ced91a8e9b`

2. NxDI doesn't natively support Qwen2_VL. To extend support, modify `/home/ubuntu/upstreaming-to-vllm/vllm/model_executor/model_loader/neuronx_distributed.py` to add support for the `Qwen2VLForConditionalGeneration` model on Neuron

Copy:
```
"MllamaForConditionalGeneration":
("neuronx_distributed_inference.models.mllama.modeling_mllama",
    "NeuronMllamaForCausalLM"),
```

Replace with:
```
"Qwen2VLForConditionalGeneration":
("neuronx_distributed_inference.models.qwen2_vl.modelling_qwen2_vl.py",
    "NeuronQwen2VLForCausalLM"),     
```

3. We include local versions of `tmp_neuronx_distributed_model_runner.py` and `tmp_neuronx_distributed.py`; replace `upstreaming-to-vllm/vllm/worker/neuronx_distributed_model_runner.py` and `upstreaming-to-vllm/vllm/model_executor/model_loader/neuronx_distributed.py` respectively with these.
This is a temporary workaround so that we don't have to maintain a local version of the `upstreaming-to-vllm` repo.

4. Run `generation_qwen2_vl.py` to get a compiled model and provide the path to the compiled model in line 13 of `vLLM_offline_qwen2_vl.py`.

### Update VLLM for qwen2

To support input padding for qwen with vllm, you will need to update `upstream-to-vllm` with the files under `qwen2-vl-neuron/examples/modified_vllm_files`. They correspond to `upstreaming-to-vllm/vllm/model_executor/models/qwen2_vl.py` and `upstreaming-to-vllm/vllm/multimodal/processing.py`.

## Launch Offline Server

Model path should point to your model downloaded from huggingface. You will need to modify `NEURON_COMPILED_ARTIFACTS` environment variable to point to your compile model's neuron artifacts.

```
chmod +x launch_offline_qwen2_vl.sh 
# Example
bash launch_offline_qwen2_vl.sh --model_path "/home/ubuntu/Qwen2-VL-7B-Instruct" --tp 4 --bs 1 --seq 4096
```

## Launch Online Server

You can start your online server by running:
```
chmod +x vllm_online_qwen2_vl.sh
bash vllm_online_qwen2_vl.sh
```

Online requests can be submitted through `curl` or through python's requests library. See `vllm_online_requests.sh` or `benchmark.py` for examples of each.

## Benchmarking
Once the vllm server is running, performance benchmarking can be run with:

```
python3 benchmark.py --model-path="/home/ubuntu/Qwen2-VL-7B-Instruct" --dataset-name="lmms-lab/DocVQA" --url http://0.0.0.0:8080/v1/chat/completions --num-examples=1
```