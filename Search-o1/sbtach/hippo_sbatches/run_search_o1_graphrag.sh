#!/bin/bash

# ================================
# Default parameters
# ================================
GRAPHRAG="graphrag"
DATASET="PopQA"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
PORT=8216
NODE_SCALE=5000
WAIT_TIME=380
SPLIT="qa_test_full"

# Paths
CONDA_ROOT="/scratch/df2362/conda_env"
CONDA_ROOT_NEW="/scratch/df2362/conda_env"
GRAPH_R1_DIR="/scratch/sl11766/My-Graph-R1/"
CLIENT_DIR="/scratch/sl11766/search-o1-project/"
LOG_DIR="/scratch/sl11766/search-o1-project/logs"

# ================================
# Command line argument parsing
# ================================
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -g, --graphrag GRAPHRAG     GraphRAG type (default: $GRAPHRAG)"
  echo "                              Supported: hypergraphrag, lightrag, hipporag, linearrag, raptor, graphrag"
  echo "  -d, --dataset DATASET       Dataset name (default: $DATASET)"
  echo "  -m, --model-path PATH       Model path (default: $MODEL_PATH)"
  echo "  -p, --port PORT             API port number (default: $PORT)"
  echo "  -s, --node-scale SCALE      Node scale (default: $NODE_SCALE)"
  echo "  -w, --wait-time SECONDS     Wait time for server startup (default: $WAIT_TIME)"
  echo "  --split SPLIT               Dataset split (default: $SPLIT)"
  echo "  -h, --help                  Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 -g graphrag -d PopQA -m 'Qwen/Qwen2.5-7B-Instruct' -p 8216"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -g|--graphrag)
      GRAPHRAG="$2"
      shift 2
      ;;
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
    -s|--node-scale)
      NODE_SCALE="$2"
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
# GraphRAG type configuration
# ================================
case "$GRAPHRAG" in
  hypergraphrag)
    API_SCRIPT="script_api_HypergraphRAG.py"
    API_ENV="graphr1"
    API_CONDA_ROOT="$CONDA_ROOT"
    CLIENT_SCRIPT="run_search_o1_hyper.py"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  lightrag)
    API_SCRIPT="script_api_LightRAG.py"
    API_ENV="lightrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    # No dedicated lightrag client script; reuse graph version by default
    CLIENT_SCRIPT="run_search_o1_graph.py"
    STARTUP_PATTERN="Application startup complete"
    ;;
  hipporag)
    API_SCRIPT="script_api_HippoRAG.py"
    API_ENV="hipporag"
    API_CONDA_ROOT="$CONDA_ROOT"
    CLIENT_SCRIPT="run_search_o1_hippo.py"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  linearrag)
    API_SCRIPT="script_api_LinearRAG.py"
    API_ENV="linearrag"
    API_CONDA_ROOT="$CONDA_ROOT_NEW"
    CLIENT_SCRIPT="run_search_o1_linear.py"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  raptor)
    API_SCRIPT="script_api_RAPTOR.py"
    API_ENV="raptor"
    API_CONDA_ROOT="$CONDA_ROOT_NEW"
    CLIENT_SCRIPT="run_search_o1_raptor.py"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  graphrag)
    API_SCRIPT="script_api_GraphRAG.py"
    API_ENV="graphrag"
    API_CONDA_ROOT="$CONDA_ROOT_NEW"
    CLIENT_SCRIPT="run_search_o1_graph.py"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  *)
    echo "Error: Unsupported GraphRAG type: $GRAPHRAG"
    echo "Supported types: hypergraphrag, lightrag, hipporag, linearrag, raptor, graphrag"
    exit 1
    ;;
esac

# ================================
# Print configuration
# ================================
echo "========================================="
echo "Search-o1 Runner Configuration"
echo "========================================="
echo "GraphRAG Type: $GRAPHRAG"
echo "Dataset: $DATASET"
echo "Model Path: $MODEL_PATH"
echo "API Port: $PORT"
echo "Node Scale: $NODE_SCALE"
echo "API Script: $API_SCRIPT"
echo "Client Script: $CLIENT_SCRIPT"
echo "API Environment: $API_ENV"
echo "Wait Time: ${WAIT_TIME}s"
echo "Split: $SPLIT"
echo "========================================="
echo ""

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Generate log file names
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')
LOG_PREFIX="${DATASET_LOWER}_${GRAPHRAG}"
SERVER_LOG="${LOG_DIR}/server_${LOG_PREFIX}_$(date +%Y%m%d_%H%M%S).log"

# ================================
# Step 1: Start Retrieval Server
# ================================
echo "=== Step 1: Starting Retrieval Server ==="
(
    # Load modules if available (for cluster environments)
    if command -v module &> /dev/null; then
        module purge 2>/dev/null || true
        module load anaconda3/2024.02 2>/dev/null || true
    fi
    
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    
    conda activate "${API_CONDA_ROOT}/${API_ENV}"
    cd "$GRAPH_R1_DIR"
    
    echo "Server working directory: $(pwd)"
    echo "Starting $API_SCRIPT on port $PORT..."
    
    python "$API_SCRIPT" \
        --data_source "$DATASET" \
        --port "$PORT" \
        --node_scale "$NODE_SCALE" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PID=$!

echo "Server process launched with PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"
echo "Waiting ${WAIT_TIME}s for server to load index..."
sleep "$WAIT_TIME"

# Check if server is still running
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

# Load modules if available
if command -v module &> /dev/null; then
    module purge 2>/dev/null || true
fi

cd "$CLIENT_DIR"

echo "Running client locally (no Singularity)..."

# Try to use conda environment if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate search_o1 2>/dev/null || {
        echo "Warning: Could not activate search_o1 conda environment"
        echo "Attempting to run with current Python environment..."
    }
fi

python "scripts/$CLIENT_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')" \
    --split "$SPLIT" \
    --bing_subscription_key 'dummy_key' \
    --bing_endpoint "http://127.0.0.1:${PORT}/search"

CLIENT_EXIT_CODE=$?

# ================================
# Cleanup
# ================================
echo ""
echo "=== Cleanup ==="
echo "Job finished. Killing server process (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true

# Wait a bit for graceful shutdown
sleep 2

# Force kill if still running
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Force killing server process..."
    kill -9 $SERVER_PID 2>/dev/null || true
fi

echo "Server process terminated."
echo "Server log saved at: $SERVER_LOG"

# Exit with client's exit code
exit $CLIENT_EXIT_CODE
