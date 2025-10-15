set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

OUTPUT_DIR="/home/ubuntu/qwen2-vl-neuron/examples/serving-logs"
LOG_FILE="vllm-serving.log"

function fail () {
  echo "$1"
  helper
  exit 1
}

function helper () {
    _cmd=$(basename "$0")
    echo ""
    echo "  [--tp | --tensor_model_parallel_size TENSOR_MODEL_PARALLEL_SIZE]  optional - the tensor model parallel size"
    echo "  [--bs | --train_batch_size TRAIN_BATCH_SIZE]  optional - the training batch size"    
    echo "  [--model_path MODEL_PATH]  optional - The directory containing NxDI-supported configs, checkpoints, and neuron compiled artifacts. "
    echo "  [--seq | --seq_length RESUME_FROM]  optional - The entire sequence length combining input and output sequence"
    echo ""
    }

# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --tp | --tensor_model_parallel_size)
            TENSOR_MODEL_PARALLEL_SIZE="--tensor_parallel_size $2"
            shift
            ;;
        --bs | --train_batch_size)
            TRAIN_BATCH_SIZE="--batch_size $2"
            shift
            ;;
        --seq | --seq_length)
            SEQUENCE_LENGTH="--seq_len $2"
            shift
            ;;
        --model_path)
            MODEL_PATH="--model_path $2"
            shift
            ;;
        -h | --help | help)
            helper
            exit 0
            ;;
        *)
            fail "Unrecognized option '$1'"
            ;;
    esac
    shift
done

#NPROC_PER_NODE=2
#DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE"

mkdir -p "$OUTPUT_DIR"

echo -e "MODEL_PATH: $MODEL_PATH, TENSOR_MODEL_PARALLEL_SIZE: $TENSOR_MODEL_PARALLEL_SIZE, TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE, \
 SEQUENCE_LENGTH: $SEQUENCE_LENGTH, TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE, OUTPUT_DIR: $OUTPUT_DIR"

python3 vLLM_offline_qwen2_vl.py \
    $MODEL_PATH $TENSOR_MODEL_PARALLEL_SIZE $TRAIN_BATCH_SIZE $SEQUENCE_LENGTH | tee $OUTPUT_DIR/$LOG_FILE