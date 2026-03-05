#!/bin/bash
#SBATCH --job-name=mus_test10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --account=pr_142_tandon_priority
#SBATCH --output=/scratch/sl11766/GraphSearch-main/logs/mus_test_%j.out
#SBATCH --error=/scratch/sl11766/GraphSearch-main/logs/mus_test_%j.err

# 定义一些路径变量 (方便修改)
PROJECT_DIR="/scratch/sl11766/GraphSearch-main"
RETRIEVER_CODE_DIR="/scratch/sl11766/My-Graph-R1"
ENV_VLLM="/scratch/sl11766/GraphSearch-main/envs"
ENV_RETRIEVER="/scratch/df2362/conda_env/graphr1"

# 【核心修复】：定义 Python 的绝对路径 (Absolute Paths)
# 这样无论 activate 是否成功，都会强制使用环境里的解释器 
PYTHON_VLLM="${ENV_VLLM}/bin/python" PYTHON_RETRIEVER="${ENV_RETRIEVER}/bin/python"

# 确保日志目录存在
mkdir -p $PROJECT_DIR/logs

echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"

module purge
module load anaconda3/2024.02
module load cuda/11.6.2 
eval "$(conda shell.bash hook)"

# ======================================================
# Step 1: 启动 vLLM 推理服务 (Port 8123)
# ======================================================
echo "=== Step 1: Starting vLLM Server (Port 8123) ==="
(
export CUDA_VISIBLE_DEVICES=0
conda activate $ENV_VLLM

python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

# 后台启动 vLLM
nohup $PYTHON_VLLM -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --trust-remote-code \
    --port 8123 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    > $PROJECT_DIR/logs/vllm_${SLURM_JOB_ID}.log 2>&1 &
)
VLLM_PID=$!
echo "vLLM launched with PID: $VLLM_PID"

# ======================================================
# Step 2: 启动 Retrieval Server (Port 8336)
# ======================================================
echo "=== Step 2: Starting Retriever Server (Port 8336) ==="
(
export CUDA_VISIBLE_DEVICES=1 
conda activate $ENV_RETRIEVER
cd $RETRIEVER_CODE_DIR
echo "Retriever working directory: $(pwd)"
echo "DEBUG Retriever: Using Python at $PYTHON_RETRIEVER"


nohup $PYTHON_RETRIEVER script_api_HypergraphRAG.py \
    --data_source Musique \
    --port 8336 \
    > $PROJECT_DIR/logs/retriever_${SLURM_JOB_ID}.log 2>&1 &
)
RETRIEVER_PID=$!
echo "Retriever launched with PID: $RETRIEVER_PID"

echo "Waiting 220s for servers to initialize..."
sleep 220

# ======================================================
# Step 3: 运行 Client Eval (Limit 10)
# ======================================================
echo "=== Step 3: Running Eval Client (Test 10 samples) ==="

conda activate $ENV_VLLM
cd $PROJECT_DIR

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行评估

python eval.py \
    --dataset musique \
    --graphrag hypergraphrag \
    --concurrency 5 \
    --limit 10

EXIT_CODE=$?

# ======================================================
# Step 4: 清理进程
# ======================================================
echo "Job finished with exit code: $EXIT_CODE"
echo "Killing background processes..."
kill $VLLM_PID
kill $RETRIEVER_PID

exit $EXIT_CODE
