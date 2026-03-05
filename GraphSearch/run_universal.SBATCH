#!/bin/bash
#SBATCH --job-name=graph_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=36:00:00      
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_142_tandon_priority
#SBATCH --output=/scratch/sl11766/GraphSearch-main/logs/main_%j.out
#SBATCH --error=/scratch/sl11766/GraphSearch-main/logs/main_%j.err

# ===============================================================
# 1. 接收外部参数 (通过 sbatch --export 传入)
# ===============================================================
# 如果没传参数，给个默认值报错提示
if [ -z "$METHOD" ] || [ -z "$DATASET" ]; then
    echo "Error: Please provide METHOD and DATASET variables."
    echo "Usage: sbatch --export=METHOD=hypergraphrag,DATASET=hotpotqa run_universal.sbatch"
    exit 1
fi
START=${START:-0}
END=${END:--1}
echo "🚀 Starting Job for METHOD: $METHOD | DATASET: $DATASET"

# ===============================================================
# 2. 核心配置：Retriever 环境与脚本的路由 (If-Elif 逻辑)
# ===============================================================
PROJECT_DIR="/scratch/sl11766/GraphSearch-main"
RETRIEVER_CODE_BASE="/scratch/sl11766/My-Graph-R1" # 5种graphrags

if [ "$METHOD" == "hypergraphrag" ]; then
    RETRIEVER_ENV_PATH="/scratch/df2362/conda_env/graphr1"
    RETRIEVER_SCRIPT="script_api_HypergraphRAG.py"
    RETRIEVER_PORT=8336  

elif [ "$METHOD" == "hipporag2" ]; then
    RETRIEVER_ENV_PATH="/scratch/df2362/conda_env/hipporag" 
    RETRIEVER_SCRIPT="script_api_HippoRAG.py"
    RETRIEVER_PORT=8316
    RETRIEVER_EXTRA_ARGS="--node_scale 5000"

elif [ "$METHOD" == "linearrag" ]; then
    RETRIEVER_ENV_PATH="/scratch/df2362/conda_env_new/linearrag" 
    RETRIEVER_SCRIPT="script_api_LinearRAG.py"
    RETRIEVER_PORT=8356

elif [ "$METHOD" == "graphrag" ]; then
    RETRIEVER_ENV_PATH="/scratch/df2362/conda_env_new/graphrag" 
    RETRIEVER_SCRIPT="script_api_GraphRAG.py"
    RETRIEVER_PORT=8326

elif [ "$METHOD" == "raptor" ]; then
    RETRIEVER_ENV_PATH="/scratch/df2362/conda_env_new/raptor" 
    RETRIEVER_SCRIPT="script_api_RAPTOR.py"
    RETRIEVER_PORT=8346


else
    echo "❌ Error: Unknown Method '$METHOD'"
    exit 1
fi

# ===============================================================
# 3. 数据集名称映射 (小写 -> CamelCase)
# ===============================================================
# eval.py 用小写 (hotpotqa)，但 Retriever 脚本通常要大写 (HotpotQA)
case "$DATASET" in
    "hotpotqa") SOURCE_NAME="HotpotQA" ;;
    "musique")  SOURCE_NAME="Musique" ;;
    "2wikimultihopqa") SOURCE_NAME="2WikiMultiHopQA" ;;
    "nq")       SOURCE_NAME="NQ" ;;
    "triviaqa") SOURCE_NAME="TriviaQA" ;;
    "popqa")    SOURCE_NAME="PopQA" ;;
    *)          SOURCE_NAME="$DATASET" ;; # 默认保持原样
esac

echo "Dataset Mapping: $DATASET -> $SOURCE_NAME"
echo "Using Retriever Env: $RETRIEVER_ENV_PATH"

# ===============================================================
# 4. 定义 Python 解释器路径
# ===============================================================
ENV_VLLM="/scratch/sl11766/GraphSearch-main/envs"
PYTHON_VLLM="${ENV_VLLM}/bin/python"
PYTHON_RETRIEVER="${RETRIEVER_ENV_PATH}/bin/python" 

# ===============================================================
# 5. 日志管理：创建分层目录
# ===============================================================
# 结构: logs/method/dataset/job_id/
LOG_DIR="$PROJECT_DIR/logs/${METHOD}/${DATASET}/${SLURM_JOB_ID}"
mkdir -p $LOG_DIR
echo "📝 Logs will be saved to: $LOG_DIR"


# ===============================================================
# Step 1: 启动 vLLM (所有任务通用)
# ===============================================================
echo "=== Step 1: Starting vLLM Server ==="
(
export CUDA_VISIBLE_DEVICES=0
# 使用 eval.py 所在环境
nohup $PYTHON_VLLM -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --trust-remote-code \
    --port 8123 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 32768 \
    > $LOG_DIR/vllm.log 2>&1 &
)
VLLM_PID=$!

# ===============================================================
# Step 2: 启动 Retriever (动态环境 + 动态脚本)
# ===============================================================
echo "=== Step 2: Starting Retriever ($METHOD) ==="
(
export CUDA_VISIBLE_DEVICES=0
#export TOKENIZERS_PARALLELISM=false
cd $RETRIEVER_CODE_BASE

# 使用对应环境的 Python
nohup $PYTHON_RETRIEVER $RETRIEVER_SCRIPT \
    --data_source $SOURCE_NAME \
    --port $RETRIEVER_PORT \
    $RETRIEVER_EXTRA_ARGS \
    > $LOG_DIR/retriever.log 2>&1 &
)
RETRIEVER_PID=$!

echo "⏳ Waiting 300s for servers to initialize..."
sleep 300
echo "=== 🔍 DEBUG: Network Diagnostic Start ==="

# 1. 检查环境变量 (看有没有 http_proxy)
echo "Checking Proxy Variables:"
env | grep -i proxy || echo "No proxy variables set."

# 2. 检查端口监听情况 (看 8316 到底是谁在听，是 127.0.0.1 还是 ::1)
echo "Checking Port 8316:"
netstat -tulnp | grep 8316 || echo "❌ No process listening on 8316!"

# 3. 尝试用 curl 详细模式连接 (模拟一次请求)
echo "Testing connection with CURL:"
curl -v http://127.0.0.1:8316/search \
     -H "Content-Type: application/json" \
     -d '{"queries": ["测试问题"], "top_k": 1}' \
     || echo "❌ CURL Connection Failed"

echo "=== 🔍 DEBUG: Network Diagnostic End ==="
# ===============================================================
# Step 3: 运行 Eval
# ===============================================================
echo "=== Step 3: Running Evaluation ==="
cd $PROJECT_DIR
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 使用 vLLM 的环境运行 eval
$PYTHON_VLLM eval.py \
    --dataset $DATASET \
    --graphrag $METHOD \
    --concurrency 1 \
    --start $START \
    --end $END \
    > $LOG_DIR/eval.log 2>&1

EXIT_CODE=$?

# ===============================================================
# Step 4: 清理
# ===============================================================
echo "Job finished with exit code: $EXIT_CODE"
kill $VLLM_PID
kill $RETRIEVER_PID
exit $EXIT_CODE
