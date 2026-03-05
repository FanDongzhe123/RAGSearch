#!/bin/bash
#SBATCH --job-name=50_dense
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=4:00:00           
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_142_tandon_priority
#SBATCH --output=/scratch/sl11766/search-o1-project/logs/50_dense_%j.out  
#SBATCH --error=/scratch/sl11766/search-o1-project/logs/50_dense_%j.err

echo "=== Step 1: Starting Retrieval Server ==="
(
    module purge
    module load anaconda3/2024.02
    eval "$(conda shell.bash hook)"
    conda activate /scratch/df2362/conda_env/retriever
    cd /scratch/sl11766/My-Search-R1/
    
    bash ANN_retrieval_launch.sh -p 8206 > ../search-o1-project/logs/ser_50_dense_%j.log 2>&1
) &
SERVER_PID=$!
echo "Server process launched with PID: $SERVER_PID"
echo "Waiting 220s for server to load index..."
sleep 220

echo "=== Step 2: Starting Client Model ==="
module purge
cd /scratch/sl11766/search-o1-project/

export JOB_ID=$SLURM_JOB_ID


singularity exec --nv \
--overlay my-overlay.ext3:ro \
my-cuda-image.sif \
/bin/bash << 'EOF'



export HF_HOME=/scratch/sl11766/hf_cache
source /ext3/env.sh
conda activate search_o1


DATASETS=("hotpotqa" "nq" "musique" "triviaqa")

for dataset in "${DATASETS[@]}"; do
    for i in {1..5}; do
        echo "--------------------------------------------------"
        echo "Starting $dataset - Run $i/5"
        echo "--------------------------------------------------"
        
  
        LOG_FILE="/scratch/sl11766/search-o1-project/logs/${dataset}_run${i}_${JOB_ID}.log"
        
    
        python scripts/run_search_o1.py \
        --model_path 'Qwen/Qwen2.5-7B-Instruct' \
        --dataset_name "$dataset" \
        --split qa_test_full \
        --subset_num 50 \
        --bing_subscription_key 'dummy_key' \
        --bing_endpoint 'http://127.0.0.1:8206/retrieve' \
        > "$LOG_FILE" 2>&1
        
        echo "Finished $dataset - Run $i. Log saved to $LOG_FILE"
        
        sleep 2
    done
done

EOF

echo "Job finished. Killing server process..."
kill $SERVER_PID
