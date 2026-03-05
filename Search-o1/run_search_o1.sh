#!/bin/bash

# ================================
# Default parameters (dense retriever + search-o1)
# ================================
DATASET="nq"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
PORT=8206
WAIT_TIME=120
SPLIT="qa_test_full"

# Paths
RETRIEVER_CONDA="./conda_env/retriever"
RETRIEVER_DIR="../Search-R1/"
CLIENT_DIR="../search-o1/"
LOG_DIR="./logs"

# ================================
# Command line argument parsing
# ================================
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -d, --dataset DATASET       Dataset name (default: $DATASET)"
  echo "  -m, --model-path PATH       Model path (default: $MODEL_PATH)"
  echo "  -p, --port PORT             API port number (default: $PORT)"
  echo "  -w, --wait-time SECONDS     Wait time for server startup (default: $WAIT_TIME)"
  echo "  --split SPLIT               Dataset split (default: $SPLIT)"
  echo "  -h, --help                  Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 -d nq -m 'Qwen/Qwen2.5-7B-Instruct' -p 8206"
  exit 1
}

while [[ $# > 0 ]]; do
  case $1 in
    -d|--dataset)
      DATASET="$2"
      shift 2
      ;;
    -m|--model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -w|--wait-time)
      WAIT_TIME="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ================================
# Print configuration
# ================================
echo "========================================="
echo "search-o1 Dense Retriever Runner"
echo "========================================="
echo "Dataset: $DATASET"
echo "Model Path: $MODEL_PATH"
echo "API Port: $PORT"
echo "Wait Time: ${WAIT_TIME}s"
echo "Split: $SPLIT"
echo "========================================="
echo ""

mkdir -p "$LOG_DIR"
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
SERVER_LOG="${LOG_DIR}/server_${DATASET_LOWER}_dense_$(date +%Y%m%d_%H%M%S).log"

# ================================
# Step 1: Start Retrieval Server
# ================================
echo "=== Step 1: Starting Retrieval Server ==="
(
    if command -v module &> /dev/null; then
        module purge 2>/dev/null || true
        module load anaconda3/2024.02 2>/dev/null || true
    fi

    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$RETRIEVER_CONDA" 2>/dev/null || {
        echo "Warning: Could not activate retriever conda env: $RETRIEVER_CONDA"
    }

    cd "$RETRIEVER_DIR" || {
        echo "Failed to cd to $RETRIEVER_DIR"
        exit 1
    }

    echo "Retriever working directory: $(pwd)"
    echo "Launching ANN_retrieval_launch.sh on port $PORT..."

    bash ANN_retrieval_launch.sh -p "$PORT" > "$SERVER_LOG" 2>&1
) &

SERVER_PID=$!
echo "Server process launched with PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"
echo "Waiting ${WAIT_TIME}s for server to load index..."
sleep "$WAIT_TIME"

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server process died unexpectedly!"
    echo "Check log file: $SERVER_LOG"
    tail -n 50 "$SERVER_LOG"
    exit 1
fi

# ================================
# Step 2: Start Client Model
# ================================
echo ""
echo "=== Step 2: Starting Client Model ==="

cd "$CLIENT_DIR" || {
  echo "Failed to cd to $CLIENT_DIR"
  kill $SERVER_PID 2>/dev/null || true
  exit 1
}

echo "Client working directory: $(pwd)"

if command -v module &> /dev/null; then
    module purge 2>/dev/null || true
fi

echo "Running client locally (no Singularity)..."

if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate search_o1 2>/dev/null || {
        echo "Warning: Could not activate search_o1 conda environment"
        echo "Attempting to run with current Python environment..."
    }
fi

python scripts/run_search_o1.py \
    --model_path "$MODEL_PATH" \
    --dataset_name "$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')" \
    --split "$SPLIT" \
    --bing_subscription_key 'dummy_key' \
    --bing_endpoint "http://127.0.0.1:${PORT}/retrieve"

CLIENT_EXIT_CODE=$?

# ================================
# Cleanup
# ================================
echo ""
echo "=== Cleanup ==="
echo "Job finished. Killing server process (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true

sleep 2

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Force killing server process..."
    kill -9 $SERVER_PID 2>/dev/null || true
fi

echo "Server process terminated."
echo "Server log saved at: $SERVER_LOG"

exit $CLIENT_EXIT_CODE

