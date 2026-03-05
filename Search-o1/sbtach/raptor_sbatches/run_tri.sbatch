#!/bin/bash
#SBATCH --job-name=tri_raptor
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_142_tandon_priority
#SBATCH --output=/scratch/sl11766/search-o1-project/logs/run_tri_raptor_%j.out
#SBATCH --error=/scratch/sl11766/search-o1-project/logs/run_tri_raptor_%j.err
echo "=== Step 1: Starting Retrieval Server ==="
(
    module purge
    module load anaconda3/2024.02
    eval "$(conda shell.bash hook)"
    
     
    conda activate /scratch/df2362/conda_env_new/raptor
    
    
    cd /scratch/sl11766/My-Graph-R1/
    
    echo "Server working directory: $(pwd)"
    
     
    python script_api_RAPTOR.py \
        --data_source TriviaQA \
        --port 8246 \
        > /scratch/sl11766/search-o1-project/logs/server_tri_raptor_${SLURM_JOB_ID}.log 2>&1
) &
 
SERVER_PID=$!
echo "Server process launched with PID: $SERVER_PID"
 
echo "Waiting 180s for server to load index..."
sleep 180
echo "=== Step 2: Starting Client Model ==="
module purge

cd /scratch/sl11766/search-o1-project/
singularity exec --nv \
--overlay my-overlay.ext3:ro \
my-cuda-image.sif \
/bin/bash -c "
export HF_HOME=/scratch/sl11766/hf_cache; 
source /ext3/env.sh; 
conda activate search_o1; 
echo 'Running Client Script...'
python scripts/run_search_o1_raptor.py \
--model_path 'Qwen/Qwen2.5-7B-Instruct' \
--dataset_name triviaqa \
--split qa_test_full \
--bing_subscription_key 'dummy_key' \
--bing_endpoint 'http://127.0.0.1:8246/search'
"
echo "Job finished. Killing server process..."
kill $SERVER_PID
