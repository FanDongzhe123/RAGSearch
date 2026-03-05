#!/bin/bash
# 只提交 HotpotQA 2000-3000 与 3000-4000 两个任务

set -e
cd "$(dirname "$0")"

echo "提交 Hippo HotpotQA 2000-3000 / 3000-4000 ..."
sbatch Hippo_HotpotQA_2000_3000.SBATCH
sbatch Hippo_HotpotQA_3000_4000.SBATCH
echo "完成 (共 2 个任务)"
