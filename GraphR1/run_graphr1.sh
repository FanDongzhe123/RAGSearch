DEFAULT_PORT=8002
API_PORT=$DEFAULT_PORT
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME="Qwen2.5-3B-Instruct"
DATASET="2WikiMultiHopQA"
NODE_SCALE=1
GRAPHRAG="hypergraphrag"
BATCH_SIZE=128
N_GPU=4

# Single conda root for all GraphRAG environments
CONDA_ROOT="../conda_env"

# ================================
# Command line argument parsing
# ================================
usage() {
  echo "Usage: $0 [-a API_PORT] [-p MODEL_PATH] [-m MODEL_NAME] [-d DATASET] [-b BATCH_SIZE] [-g N_GPU] [-s NODE_SCALE] [-r GRAPHRAG]"
  echo "  -a: API service starting port (default: $DEFAULT_PORT)"
  echo "  -p: Model path"
  echo "  -m: Model name"
  echo "  -d: Dataset name (supports mixed: NQ-HotpotQA)"
  echo "  -b: Batch size (default: 128)"
  echo "  -g: Number of GPUs (default: 4)"
  echo "  -s: Node scale (default: 1)"
  echo "  -r: GraphRAG type (hypergraphrag/lightrag/hipporag/linearrag/raptor/graphrag, default: hypergraphrag)"
  exit 1
}

while getopts "a:p:m:d:b:g:s:r:h" opt; do
  case $opt in
    a) API_PORT=$OPTARG ;;
    p) MODEL_PATH=$OPTARG ;;
    m) MODEL_NAME=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    b) BATCH_SIZE=$OPTARG ;;
    g) N_GPU=$OPTARG ;;
    s) NODE_SCALE=$OPTARG ;;
    r) GRAPHRAG=$OPTARG ;;
    h) usage ;;
    *) usage ;;
  esac
done

# ================================
# Environment and script mapping
# ================================
case "$GRAPHRAG" in
  hypergraphrag)
    API_SCRIPT="script_api_HypergraphRAG.py"
    API_ENV="graphr1"
    API_CONDA_ROOT="$CONDA_ROOT"
    TRAIN_SCRIPT="run_grpo_hypergraphrag.sh"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  lightrag)
    API_SCRIPT="script_api_LightRAG.py"
    API_ENV="lightrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    TRAIN_SCRIPT="run_grpo_lightrag.sh"
    STARTUP_PATTERN="Application startup complete"
    ;;
  hipporag)
    API_SCRIPT="script_api_HippoRAG.py"
    API_ENV="hipporag"
    API_CONDA_ROOT="$CONDA_ROOT"
    TRAIN_SCRIPT="run_grpo_hipporag.sh"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  linearrag)
    API_SCRIPT="script_api_LinearRAG.py"
    API_ENV="linearrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    TRAIN_SCRIPT="run_grpo_linearrag.sh"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  raptor)
    API_SCRIPT="script_api_RAPTOR.py"
    API_ENV="raptor"
    API_CONDA_ROOT="$CONDA_ROOT"
    TRAIN_SCRIPT="run_grpo_raptor.sh"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  graphrag)
    API_SCRIPT="script_api_GraphRAG.py"
    API_ENV="graphrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    TRAIN_SCRIPT="run_grpo_graphrag.sh"
    STARTUP_PATTERN="Uvicorn running on"
    ;;
  *)
    echo "Error: Unsupported GraphRAG type: $GRAPHRAG"
    echo "Supported types: hypergraphrag, lightrag, hipporag, linearrag, raptor, graphrag"
    exit 1
    ;;
esac

TRAIN_ENV="graphr1"

# Generate config file name
CONFIG_NAME="config_${DATASET}_${GRAPHRAG}_scale${NODE_SCALE}.json"

# ================================
# Print configuration summary
# ================================
echo "========================================="
echo "Graph-R1 Training Startup Script"
echo "========================================="
echo "GraphRAG Type: $GRAPHRAG"
echo "API Script: $API_SCRIPT"
echo "API Environment: $API_ENV"
echo "API Environment Path: $API_CONDA_ROOT/$API_ENV"
echo "Training Script: $TRAIN_SCRIPT"
echo "Training Environment: $TRAIN_ENV"
echo "Training Environment Path: $CONDA_ROOT/$TRAIN_ENV"
echo "Dataset: $DATASET"
echo "Starting Port: $API_PORT"
echo "Model Path: $MODEL_PATH"
echo "Model Name: $MODEL_NAME"
echo "Node Scale: $NODE_SCALE"
echo "Batch Size: $BATCH_SIZE"
echo "Number of GPUs: $N_GPU"
echo "Config File: $CONFIG_NAME"
echo "========================================="
echo ""

# ================================
# Dataset parsing
# ================================
parse_datasets() {
  echo "$1" | sed 's/[-+]/ /g'
}

DATASET_LIST=($(parse_datasets "$DATASET"))
echo "Detected ${#DATASET_LIST[@]} dataset(s): ${DATASET_LIST[@]}"
echo ""

# ================================
# Update port configuration
# ================================
echo ">>> Updating port configuration"
python update_port_config.py \
  --port $API_PORT \
  --dataset "$DATASET" \
  --graphrag "$GRAPHRAG" \
  --node_scale $NODE_SCALE

if [ $? -ne 0 ]; then
  echo "✗ Failed to update port configuration"
  exit 1
fi
echo ""

# ================================
# Export environment variables for SearchTool
# ================================
export DATASET="$DATASET"
export GRAPHRAG="$GRAPHRAG"
export NODE_SCALE="$NODE_SCALE"
export CONFIG_NAME="$CONFIG_NAME"

echo ">>> Setting environment variables"
echo "  DATASET=$DATASET"
echo "  GRAPHRAG=$GRAPHRAG"
echo "  NODE_SCALE=$NODE_SCALE"
echo "  CONFIG_NAME=$CONFIG_NAME"
echo ""

# ================================
# Start API services
# ================================
if [ ${#DATASET_LIST[@]} -gt 1 ]; then
  echo ">>> Starting API services in multi-dataset mode"
  CURRENT_PORT=$API_PORT
  declare -a API_PIDS
  
  for DATASET_NAME in "${DATASET_LIST[@]}"; do
    API_PYTHON="${API_CONDA_ROOT}/${API_ENV}/bin/python"
    LOG_FILE="result_api_${GRAPHRAG}_${DATASET_NAME}_scale${NODE_SCALE}_port${CURRENT_PORT}.log"
    
    echo "  Starting [$DATASET_NAME] API service"
    echo "    GraphRAG: $GRAPHRAG"
    echo "    Python: $API_PYTHON"
    echo "    Port: $CURRENT_PORT"
    echo "    Node Scale: $NODE_SCALE"
    echo "    Log: $LOG_FILE"
    
    nohup $API_PYTHON -u $API_SCRIPT \
      --data_source $DATASET_NAME \
      --port $CURRENT_PORT \
      --node_scale $NODE_SCALE \
      > $LOG_FILE 2>&1 &
    
    API_PIDS+=($!)
    echo "    PID: ${API_PIDS[-1]}"
    echo ""
    CURRENT_PORT=$((CURRENT_PORT + 1))
  done

  echo ">>> Waiting for all API services to start..."
  CURRENT_PORT=$API_PORT
  
  for i in "${!DATASET_LIST[@]}"; do
    DATASET_NAME="${DATASET_LIST[$i]}"
    LOG_FILE="result_api_${GRAPHRAG}_${DATASET_NAME}_scale${NODE_SCALE}_port${CURRENT_PORT}.log"
    echo -n "  Waiting for [$DATASET_NAME] (port $CURRENT_PORT): "
    
    WAIT_TIMEOUT=30000
    WAIT_COUNT=0
    
    while [ ! -f "$LOG_FILE" ] || ! grep -q "$STARTUP_PATTERN" "$LOG_FILE" 2>/dev/null; do
      # Check if process is still running
      if [ ! -z "${API_PIDS[$i]}" ]; then
        if ! ps -p ${API_PIDS[$i]} > /dev/null 2>&1; then
          echo " ✗ (Process exited abnormally)"
          echo "    Log: $LOG_FILE"
          tail -n 20 "$LOG_FILE" 2>/dev/null | sed 's/^/      /'
          break
        fi
      fi
      
      if [ $WAIT_COUNT -ge $WAIT_TIMEOUT ]; then
        echo " ✗ (Startup timeout)"
        echo "    Last 20 lines of log:"
        tail -n 20 "$LOG_FILE" 2>/dev/null | sed 's/^/      /'
        break
      fi
      
      sleep 1
      echo -n "."
      WAIT_COUNT=$((WAIT_COUNT+1))
    done
    
    if grep -q "$STARTUP_PATTERN" "$LOG_FILE" 2>/dev/null; then
      echo " ✓"
    fi
    CURRENT_PORT=$((CURRENT_PORT + 1))
  done

else
  echo ">>> Starting single-dataset API service"
  API_PYTHON="${API_CONDA_ROOT}/${API_ENV}/bin/python"
  LOG_FILE="result_api_${GRAPHRAG}_${DATASET}_scale${NODE_SCALE}_port${API_PORT}.log"
  
  echo "  Dataset: $DATASET"
  echo "  GraphRAG: $GRAPHRAG"
  echo "  Python: $API_PYTHON"
  echo "  Port: $API_PORT"
  echo "  Node Scale: $NODE_SCALE"
  echo "  Log: $LOG_FILE"
  
  nohup $API_PYTHON -u $API_SCRIPT \
    --data_source $DATASET \
    --port $API_PORT \
    --node_scale $NODE_SCALE \
    > $LOG_FILE 2>&1 &
  API_PID=$!
  echo "  PID: $API_PID"
  echo ""
  
  WAIT_TIMEOUT=30000
  WAIT_COUNT=0
  echo -n "  Waiting for API service to start: "
  
  while ! grep -q "$STARTUP_PATTERN" "$LOG_FILE" 2>/dev/null; do
    if ! ps -p $API_PID > /dev/null 2>&1; then
      echo " ✗ (Process exited abnormally)"
      echo "    Log: $LOG_FILE"
      tail -n 20 "$LOG_FILE" 2>/dev/null | sed 's/^/      /'
      exit 1
    fi
    if [ $WAIT_COUNT -ge $WAIT_TIMEOUT ]; then
      echo " ✗ (Startup timeout)"
      echo "    Last 20 lines of log:"
      tail -n 20 "$LOG_FILE" 2>/dev/null | sed 's/^/      /'
      exit 1
    fi
    sleep 1
    echo -n "."
    WAIT_COUNT=$((WAIT_COUNT+1))
  done
  echo " ✓"
fi

echo ""
echo "========================================="
echo "All API services started successfully"
echo "========================================="
echo ""

# ================================
# Run training
# ================================
TRAIN_PYTHON="${CONDA_ROOT}/${TRAIN_ENV}/bin/python"
echo ">>> Starting training"
echo "  Training script: $TRAIN_SCRIPT"
echo "  Training environment: $TRAIN_ENV"
echo "  Training Python: $TRAIN_PYTHON"
echo "  Config file: $CONFIG_NAME"
echo ""

bash $TRAIN_SCRIPT \
  -p "$MODEL_PATH" \
  -m "$MODEL_NAME" \
  -d "$DATASET" \
  -b "$BATCH_SIZE" \
  -g "$N_GPU" \
  -s "$NODE_SCALE" \
  -c "$CONFIG_NAME"

TRAIN_EXIT_CODE=$?

# ================================
# Cleanup API services
# ================================
echo ""
echo "========================================="
echo "Training completed, shutting down API services"
echo "========================================="

if [ ${#DATASET_LIST[@]} -gt 1 ]; then
  CURRENT_PORT=$API_PORT
  for DATASET_NAME in "${DATASET_LIST[@]}"; do
    echo "  Shutting down [$DATASET_NAME] API (port: $CURRENT_PORT)"
    fuser -k ${CURRENT_PORT}/tcp 2>/dev/null
    CURRENT_PORT=$((CURRENT_PORT + 1))
  done
else
  echo "  Shutting down [$DATASET] API (port: $API_PORT)"
  fuser -k $API_PORT/tcp 2>/dev/null
fi

echo ""
echo "========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
  echo "✅ All tasks completed"
else
  echo "⚠️  Training failed (exit code: $TRAIN_EXIT_CODE)"
fi
echo "========================================="

exit $TRAIN_EXIT_CODE