import argparse
import asyncio
import json
import os
import re
import string
import logging
import time
import collections
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
# import random
# import numpy as np


from pipeline import initialize_grag, graph_search_reasoning, naive_grag_reasoning, naive_rag_reasoning
import pipeline
pipeline.grag_method = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_PATHS = {
    "musique": "/scratch/sl11766/My-Search-R1/datasets/musique/qa_test_full.json",
    "hotpotqa": "/scratch/sl11766/My-Search-R1/datasets/hotpotqa/qa_test_full.json", 
    "2wikimultihopqa": "/scratch/sl11766/My-Search-R1/datasets/2wikimultihopqa/qa_test_full.json", 
    "nq": "/scratch/sl11766/My-Search-R1/datasets/nq/qa_test_full.json", 
    "triviaqa": "/scratch/sl11766/My-Search-R1/datasets/triviaqa/qa_test_full.json",
    "popqa": "/scratch/sl11766/My-Search-R1/datasets/popqa/qa_test_full.json" 
}
# =================配置输出目录=================
# 所有的 Outputs 和 Logs 将会被存放在这里
LOG_DIR = "./logs"
OUTPUT_DIR = "./outputs"

# 确保目录存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ============================================

# 配置 Log 文件名 (包含时间戳)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(LOG_DIR, f"eval_run_{current_time}.log")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'), # 写入文件
        logging.StreamHandler() # 输出到控制台
    ]
)

def extract_final_answer(text):
    """从长文本中提取 **Final Answer:** 之后的内容"""
    if not isinstance(text, str):
        return str(text)
    
    # 常见的几种 Final Answer 标记格式
    # patterns = [
    #     r"\*\*Final Answer:\*\*\s*(.*)",  # Bold markdown
    #     r"Final Answer:\s*(.*)",          # Plain text
    #     r"ANSWER:\s*(.*)"                 # Uppercase
    # ]
    patterns = [
        # 1. 匹配 Markdown 标题 (### Final Answer) - 允许有或没有冒号
        r"###\s*Final\s*Answer\s*:?\s*(.*)", 
        
        # 2. 匹配加粗格式 (**Final Answer**) - 允许有或没有冒号
        r"\*\*Final\s*Answer\*\*\s*:?\s*(.*)",
              
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 如果找不到标记，回退策略：
    # 直接返回原文本
    return text

# ==========================================
# 1. Metric Functions 
# ==========================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_contain_score(prediction, ground_truth):
    return normalize_answer(ground_truth) in normalize_answer(prediction)

def get_metrics(prediction, ground_truths):
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
        
    em_score = max((exact_match_score(prediction, gt) for gt in ground_truths), default=0)
    f1_score = max((compute_f1_score(prediction, gt) for gt in ground_truths), default=0)
    contain_score = max((compute_contain_score(prediction, gt) for gt in ground_truths), default=0)
    
    return {
        "em": em_score,
        "f1": f1_score,
        "contain": contain_score
    }

# ==========================================
# 2. 数据处理辅助函数
# ==========================================

def get_ground_truth(item):
    """兼容各种的 JSON 键名，并统一返回 List[str]"""
    keys_to_check = ["answer", "Answer", "answers", "golden_answers", "possible_answers"]
    
    val = None
    for key in keys_to_check:
        if key in item:
            val = item[key]
            break
            
    if val is None:
        return []

    # 如果是字符串，转列表
    if isinstance(val, str):
        return [val]
    # 如果是列表，确保里面都是字符串
    elif isinstance(val, list):
        return [str(v) for v in val]
    return []

# ==========================================
# 3. 核心异步处理逻辑
# ==========================================

async def process_single_item(sem, item, args):
    """处理单条数据：推理 -> 计算指标 -> 返回结果"""
    async with sem: 
        question = item.get('question', item.get('Question'))
        ground_truths = get_ground_truth(item)
        
        try:
            start_time = time.time()
            retrieval_count = 0
            if args.method == "graphsearch":
                # 这里调用的是 pipeline 里的异步函数
                prediction, retrieval_count= await graph_search_reasoning(question)
            elif args.method == "naive":
                prediction, retrieval_count = await naive_grag_reasoning(question)
            elif args.method == "dense":
                # Dense RAG: 远程 retrieve + LLM 生成 (Search-R1 风格)
                prediction, retrieval_count = await naive_rag_reasoning(question, top_k=args.top_k)
            else:
                prediction = ""
                retrieval_count=0
            end_time = time.time()
            
            
            full_response = prediction
            # 提取 Final Answer 用于评分
            extracted_answer = extract_final_answer(full_response)
            metrics = get_metrics(extracted_answer, ground_truths)
            # 计算指标 
            #metrics = get_metrics(prediction, ground_truths)
            
            return {
                "question": question,
                "ground_truth": ground_truths,
                "prediction": full_response,
                "extracted_answer": extracted_answer,
                "metrics": metrics,
                "retrieval_count": retrieval_count,
                "latency": round(end_time - start_time, 2)
            }
        except Exception as e:
            logging.error(f"Error processing question: {question[:30]}... Error: {str(e)}")
            return {
                "question": question,
                "ground_truths": ground_truths,
                "prediction": f"ERROR: {str(e)}",
                "extracted_answer":f"ERROR: {str(e)}",
                "metrics": {"em": 0, "f1": 0, "contain": 0},
                "retrieval_count": 0,
                "latency": 0
            }


async def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Eval Script for GraphSearch")
    
    # 数据集选择
    parser.add_argument("-d", "--dataset", required=True, 
                        choices=["hotpotqa", "musique", "2wikimultihopqa", "nq", "triviaqa", "popqa"],
                        help="Name of the dataset")
    
    # 数据集路径
    parser.add_argument("--data_path", default=None, help="Custom path to json dataset file")

    # 检索模式 
    parser.add_argument("-g", "--graphrag", required=True,
                        help="Method name (e.g., raptor, hypergraphrag, hipporag2)")
    
    # 推理模式
    parser.add_argument("-m", "--method", default="graphsearch", 
                        choices=["graphsearch", "naive", "dense"], 
                        help="Reasoning pipeline type (dense=remote RAG retrieve + LLM)")
    # 新增参数
    #parser.add_argument("--runs", type=int, default=1, help="Run the evaluation N times with random sampling")
    #parser.add_argument("--sample_size", type=int, default=-1, help="Number of samples per run (e.g. 1000)")
        # 其他参数
    parser.add_argument("--start", type=int, default=0, help="Start index of dataset")
    parser.add_argument("--end", type=int, default=-1, help="End index of dataset (-1 for all)")
    # limit 参数保留，但逻辑稍作调整，或者你可以直接忽略它
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("-k", "--top_k", default=5, type=int)
    parser.add_argument("--limit", type=int, default=-1, help="Test only N samples (debug use)")
    
    args = parser.parse_args()
    
    logging.info(f"Initializing GraphRAG Client: {args.graphrag}...")
    # 初始化客户端对象
    client_instance = initialize_grag(grag_name=args.graphrag, top_k=args.top_k, dataset=args.dataset)
    
    # 将初始化的 client 赋值给 pipeline 模块的全局变量
    pipeline.grag_method = client_instance

    if args.dataset in DATASET_PATHS:
        file_path = DATASET_PATHS[args.dataset]
        logging.info(f"Selected dataset '{args.dataset}' from preset paths.")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # ================= 修改 2: 实现切片逻辑 =================
    total_len = len(data)
    start_idx = args.start

    # 如果 end 是 -1，则取到最后；否则取 args.end；end 大于总数时等于总数
    if args.end == -1:
        end_idx = total_len
    else:
        end_idx = args.end
        if end_idx > total_len:
            logging.info(f"end ({end_idx}) > total ({total_len}), using end_idx = {total_len}")
            end_idx = total_len

    # 防止索引越界
    start_idx = max(0, start_idx)
    end_idx = min(total_len, end_idx)

    # 执行切片
    data = data[start_idx : end_idx]
    logging.info(f"Dataset Range: [{start_idx} : {end_idx}]. Total samples to test: {len(data)}")
    
    if args.limit:
        data = data[:args.limit]
        logging.info(f"Limit set: Testing only first {args.limit} samples")
    # 用于存储每一轮的平均分
    # history_metrics = {
    #     "em": [], "f1": [], "contain": [], "latency": [], "retrievals": []
    # }
    
    # total_runs = args.runs
    # sample_size = args.sample_size
    
    # print(f"\n Starting Evaluation: {total_runs} Runs | Sample Size: {sample_size if sample_size > 0 else 'ALL'}\n")

    # for run_i in range(total_runs):
    #     logging.info(f"=== RUN {run_i + 1} / {total_runs} ===")
        
    #     if sample_size > 0 and sample_size < len(full_data):
    #         # 随机抽取 sample_size 个
    #         current_data = random.sample(full_data, sample_size)
    #     else:
    #         # 如果没设置采样，或者 limit 生效 (保持兼容性)
    #         current_data = full_data[:args.limit] if args.limit > 0 else full_data

    sem = asyncio.Semaphore(args.concurrency)
    logging.info(f"Starting evaluation with concurrency={args.concurrency}...")
    
    tasks = [process_single_item(sem, item, args) for item in data]
    
    # 使用 tqdm 显示进度
    results = await tqdm_asyncio.gather(*tasks)

    total_samples = len(results)
    if total_samples > 0:
        # 计算平均 F1
        avg_f1 = sum(item['metrics']['f1'] for item in results) / total_samples
        # 计算平均 EM (Exact Match) 
        avg_em = sum(1 for item in results if item['metrics']['em']) / total_samples
        avg_contain = sum(1 for item in results if item['metrics']['contain']) / total_samples
        # 计算平均延迟 (Latency)
        avg_latency = sum(item['latency'] for item in results) / total_samples
        # 计算平均检索次数 (Retrieval Count)
        avg_retrievals = sum(item.get('retrieval_count', 0) for item in results) / total_samples
    else:
        avg_f1 = avg_em =avg_contain= avg_latency = avg_retrievals = 0.0
    

    print(f"Evaluation Report: {args.dataset} / {args.graphrag}")
    print(f"   Samples: {len(results)}")
    print(f"   Range: {start_idx} - {end_idx}")
    print(f"   Avg EM:        {avg_em:.4f}")
    print(f"   Avg Contain:    {avg_contain:.4f}")
    print(f"   Avg F1:        {avg_f1:.4f}")
    print(f"   Avg Latency:   {avg_latency:.2f}s")
    print(f"   Avg Retrievals: {avg_retrievals:.2f}")

    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_suffix = f"{args.dataset}_{args.graphrag}_range_{start_idx}_{end_idx}_{timestamp}"

    predictions_filename = os.path.join(
        OUTPUT_DIR, 
        f"predictions_{file_suffix}.json"
    )

    predictions_data = {
        "meta": {
            "dataset": args.dataset,
            "method": args.method,
            "graphrag": args.graphrag,
            "timestamp": timestamp,
            "total_samples": total_samples,
            "range_start": start_idx, # 记录元数据
            "range_end": end_idx
        },
        "details": results # 这里是原本的列表：包含 question, prediction, truth, 单个metrics等
    }

    # output_filename = os.path.join(
    #     OUTPUT_DIR, 
    #     f"eval_{args.dataset}_{args.graphrag}_{current_time}.json"
    # )

    # output_data = {
    #     "meta": {
    #         "dataset": args.dataset,
    #         "method": args.method,
    #         "graphrag": args.graphrag,
    #         "timestamp": timestamp,
    #     },
    #     "results": results  
    # }
    with open(predictions_filename, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    print(f"Detailed predictions saved to: {predictions_filename}")

    # with open(output_filename, 'w', encoding='utf-8') as f:
    #     json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # logging.info(f"Detailed results saved to: {output_filename}")

    metrics_filename = os.path.join(
        OUTPUT_DIR, 
        f"metrics_{file_suffix}.json"
    )

    metrics_data = {
        "dataset": args.dataset,
        "method": args.method,
        "graphrag": args.graphrag,
        "timestamp": timestamp,
        "num_samples": total_samples,
        "range_start": start_idx,
        "range_end": end_idx,
        "metrics": {
            "avg_em": avg_em,
            "avg_contain": avg_contain,
            "avg_f1": avg_f1,
            "avg_latency": avg_latency,
            "avg_retrieval_count": avg_retrievals
        }
    }

    with open(metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)
    print(f"Summary metrics saved to: {metrics_filename}")


if __name__ == "__main__":
    asyncio.run(main())