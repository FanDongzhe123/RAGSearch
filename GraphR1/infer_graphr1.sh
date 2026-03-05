#!/bin/bash

###############################################
# Default parameters
###############################################

DEFAULT_PORT=8170
API_PORT=$DEFAULT_PORT
MODEL_BASE="./checkpoints/Graph-R1"
MODEL_NAME="Qwen2.5-3B-Instruct"

DATASET="NQ"
NODE_SCALE=5000
GRAPHRAG="hipporag"
BATCH_SIZE=""
N_GPU=2

# Single conda root for all GraphRAG environments
CONDA_ROOT="../conda_env"

###############################################
# Print usage
###############################################
usage() {
  echo "Usage: $0 [-a API_PORT] [-d DATASET] [-s NODE_SCALE] [-r GRAPHRAG] [-b BATCH_SIZE] [-g N_GPU]"
  exit 1
}

###############################################
# Parameter parsing
###############################################
while getopts "a:d:s:r:b:g:h" opt; do
  case $opt in
    a) API_PORT=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    s) NODE_SCALE=$OPTARG ;;
    r) GRAPHRAG=$OPTARG ;;
    b) BATCH_SIZE=$OPTARG ;;
    g) N_GPU=$OPTARG ;;
    h) usage ;;
    *) usage ;;
  esac
done

###############################################
# GraphRAG type configuration
###############################################
case "$GRAPHRAG" in
  hipporag)
    API_SCRIPT="script_api_HippoRAG.py"
    API_ENV="hipporag"
    API_CONDA_ROOT="$CONDA_ROOT"
    STARTUP_PATTERN="Uvicorn running on"
    DEFAULT_BS=32
    ;;

  linearrag)
    API_SCRIPT="script_api_LinearRAG.py"
    API_ENV="linearrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    STARTUP_PATTERN="Uvicorn running on"
    DEFAULT_BS=32
    ;;

  lightrag)
    API_SCRIPT="script_api_LightRAG.py"
    API_ENV="lightrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    STARTUP_PATTERN="Application startup complete"
    DEFAULT_BS=32
    ;;

  raptor)
    API_SCRIPT="script_api_RAPTOR.py"
    API_ENV="raptor"
    API_CONDA_ROOT="$CONDA_ROOT"
    STARTUP_PATTERN="Uvicorn running on"
    DEFAULT_BS=128
    ;;

  hypergraphrag)
    API_SCRIPT="script_api_HypergraphRAG.py"
    API_ENV="graphr1"
    API_CONDA_ROOT="$CONDA_ROOT"
    STARTUP_PATTERN="Uvicorn running on"
    DEFAULT_BS=128
    ;;

  graphrag)
    API_SCRIPT="script_api_GraphRAG.py"
    API_ENV="graphrag"
    API_CONDA_ROOT="$CONDA_ROOT"
    STARTUP_PATTERN="Uvicorn running on"
    DEFAULT_BS=128
    ;;

  *)
    echo "❌ Error: Unsupported GraphRAG type: $GRAPHRAG"
    exit 1
    ;;
esac

###############################################
# Set final batch size
###############################################
if [ -z "$BATCH_SIZE" ]; then
  BATCH_SIZE=$DEFAULT_BS
fi

###############################################
# Auto-generate MODEL_PATH
###############################################
MODEL_PATH="${MODEL_BASE}/${MODEL_NAME}_${DATASET}_grpo_${NODE_SCALE}_${GRAPHRAG}_${BATCH_SIZE}/model"

###############################################
# GPU settings: CUDA_VISIBLE_DEVICES
###############################################
CUDA_IDS=$(seq -s, 0 $((N_GPU-1)))

###############################################
# Print final configuration
###############################################
echo "=========================================="
echo "GraphRAG Runner"
echo "=========================================="
echo "API_PORT      = $API_PORT"
echo "MODEL_PORT    = $((API_PORT+1))"
echo "DATASET       = $DATASET"
echo "NODE_SCALE    = $NODE_SCALE"
echo "GRAPHRAG      = $GRAPHRAG"
echo "API_SCRIPT    = $API_SCRIPT"
echo "API_ENV       = $API_ENV"
echo "CUDA_DEVICES  = $CUDA_IDS"
echo "BATCH_SIZE    = $BATCH_SIZE"
echo "MODEL_PATH    = $MODEL_PATH"
echo "=========================================="

###############################################
# Start VLLM
###############################################
MODEL_PORT=$((API_PORT+1))
VLLM_LOG="result_modelapi_${MODEL_NAME}_${DATASET}_${GRAPHRAG}.log"

echo ">>> Starting VLLM (port: $MODEL_PORT)"
CUDA_VISIBLE_DEVICES=$CUDA_IDS nohup vllm serve $MODEL_PATH \
  --served-model-name agent \
  --tensor-parallel-size $N_GPU \
  --port $MODEL_PORT \
  > $VLLM_LOG 2>&1 &

###############################################
# Start GraphRAG API
###############################################
API_PY="${API_CONDA_ROOT}/${API_ENV}/bin/python"
API_LOG="result_api_${GRAPHRAG}_${DATASET}.log"

echo ">>> Starting GraphRAG API (port: $API_PORT)"
nohup $API_PY -u $API_SCRIPT \
  --data_source $DATASET \
  --port $API_PORT \
  --node_scale $NODE_SCALE \
  > $API_LOG 2>&1 &

API_PID=$!

###############################################
# Wait for API to start
###############################################
echo -n ">>> Waiting for API service to start..."

TIMEOUT=30000
COUNT=0

while ! grep -q "$STARTUP_PATTERN" "$API_LOG" 2>/dev/null; do
  sleep 1
  COUNT=$((COUNT+1))
  echo -n "."
  if [ $COUNT -ge $TIMEOUT ]; then
    echo "❌ API startup timeout"
    exit 1
  fi
done

echo " ✓ Started successfully!"

###############################################
# Run inference
###############################################
echo ">>> Running inference..."
python3 agent/vllm_infer/run.py \
  --api-base "http://localhost:${MODEL_PORT}/v1" \
  --dataset $DATASET \
  --port_config "/scratch/df2362/Graph-R1/config_${DATASET}_${GRAPHRAG}_scale${NODE_SCALE}.json" \
  --output_dir "${DATASET}_test_${GRAPHRAG}_Full.jsonl"

###############################################
# Calculate score
###############################################
echo ">>> Calculating transfer score..."
python3 evaluation/get_transfer_score.py \
  --dir /scratch/df2362/Graph-R1/agent/vllm_infer \
  --input_file "${DATASET}_test_${GRAPHRAG}_Full.jsonl"

echo ">>> Completed!"


