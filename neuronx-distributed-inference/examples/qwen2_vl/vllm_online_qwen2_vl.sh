export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/qwenvl/qwen-vl-main/neuronx-distributed-inference/examples/qwen2_vl/traced_models/seq_len_8192-image_2/" 

SEQ_LEN=8192
TP_DEGREE=2
export LOCAL_WORLD_SIZE=$TP_DEGREE


VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' \
python -m vllm.entrypoints.openai.api_server     \
--model="/home/ubuntu/qwenvl/qwen-vl-main/neuronx-distributed-inference/examples/qwen2_vl/models/Qwen/Qwen2-VL-7B-Instruct"     \
--max-num-seqs=1    \
--max-model-len=$SEQ_LEN     \
--tensor-parallel-size=$TP_DEGREE     \
--port=8080     \
--limit-mm-per-prompt image=2 \
--device "neuron"     \
--override-neuron-config "{\"enable_bucketing\":false}"
