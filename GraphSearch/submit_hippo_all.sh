#!/bin/bash
# 依次提交所有 Hippo (hipporag2) 评估任务
# 使用方式: cd /scratch/df2362/GraphSearch-main && bash submit_hippo_all.sh

set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "依次提交 Hippo SBATCH 任务"
echo "=========================================="

# 0-1000
sbatch Hippo_HotpotQA_0_1000.SBATCH
sbatch Hippo_NQ_0_1000.SBATCH
sbatch Hippo_Musique_0_1000.SBATCH
sbatch Hippo_PopQA_0_1000.SBATCH
sbatch Hippo_TriviaQA_0_1000.SBATCH
sbatch Hippo_2WikiMultiHopQA_0_1000.SBATCH

# 1000-2000
sbatch Hippo_HotpotQA_1000_2000.SBATCH
sbatch Hippo_NQ_1000_2000.SBATCH
sbatch Hippo_Musique_1000_2000.SBATCH
sbatch Hippo_PopQA_1000_2000.SBATCH
sbatch Hippo_TriviaQA_1000_2000.SBATCH
sbatch Hippo_2WikiMultiHopQA_1000_2000.SBATCH

# 2000-3000
sbatch Hippo_HotpotQA_2000_3000.SBATCH
sbatch Hippo_NQ_2000_3000.SBATCH
sbatch Hippo_Musique_2000_3000.SBATCH
sbatch Hippo_PopQA_2000_3000.SBATCH
sbatch Hippo_TriviaQA_2000_3000.SBATCH
sbatch Hippo_2WikiMultiHopQA_2000_3000.SBATCH

# 3000-4000 (无 NQ/Musique)
sbatch Hippo_HotpotQA_3000_4000.SBATCH
sbatch Hippo_PopQA_3000_4000.SBATCH
sbatch Hippo_TriviaQA_3000_4000.SBATCH
sbatch Hippo_2WikiMultiHopQA_3000_4000.SBATCH

echo "=========================================="
echo "全部提交完成 (共 22 个任务)"
echo "=========================================="
