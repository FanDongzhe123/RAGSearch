# import json
# import re
# import argparse
# import requests
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# import torch
# import pdb


# def get_query(text):
#     pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
#     matches = pattern.findall(text)
#     return matches[-1] if matches else None


# def search(query: str):
#     payload = {"queries": [query], "topk": 3, "return_scores": True}
#     results = requests.post("http://127.0.0.1:8205/retrieve", json=payload).json()['result']

#     def _passages2string(retrieval_result):
#         format_reference = ''
#         for idx, doc_item in enumerate(retrieval_result):
#             content = doc_item['document']['contents']
#             title = content.split("\n")[0]
#             text = "\n".join(content.split("\n")[1:])
#             format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
#         return format_reference

#     return _passages2string(results[0])


# def run_reasoning(question, llm, curr_search_template, max_turns):
#     if question[-1] != "?":
#         question += "?"
#     prompt = f"""Answer the given question. \
# You must conduct reasoning inside <think> and </think> first every time you get new information. \
# After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
# You can search as many times as your want. \
# If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

#     cnt = 0
#     target_sequences = [
#         "</search>", " </search>", "</search>\n", " </search>\n",
#         "</search>\n\n", " </search>\n\n"
#     ]
#     while True:
#         sampling_params = SamplingParams(
#             temperature=0.7,
#             max_tokens=1024,
#             stop=target_sequences,
#         )
        

#         outputs = llm.generate([prompt], sampling_params)
#         output_text = outputs[0].outputs[0].text
#         if any(s in output_text for s in target_sequences):
#             # 已经生成并包含 stop token（几率小）
#             pass
#         else:
#             # vLLM 截掉了，手动拼回最后一个 stop
#             output_text += target_sequences[0]

#         # 如果生成里包含最终答案
#         if "</answer>" in output_text:
#             prompt += output_text
#             break

#         tmp_query = get_query(output_text)
#         if tmp_query:
#             search_results = search(tmp_query)
#         else:
#             search_results = ''

#         search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
#         prompt += search_text
#         cnt += 1

#         if cnt >= max_turns:
#             prompt += "\n<answer> Unable to complete reasoning within max turns. </answer>\n"
#             break

#     return prompt, cnt


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_id", type=str, required=True, help="Model ID for vLLM model")
#     parser.add_argument("--dataset_path", type=str, required=True, help="Path to input dataset (jsonl)")
#     parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions (json)")
#     parser.add_argument("--max_turns", type=int, default=5, help="Maximum number of reasoning-search turns")
#     parser.add_argument("--tensor_parallel_size", type=int, default=torch.cuda.device_count(), 
#                         help="Number of GPUs for tensor parallelism")
#     args = parser.parse_args()

#     # 多 GPU 加载模型
#     llm = LLM(
#         model=args.model_id,
#         dtype="float16",
#         tensor_parallel_size=args.tensor_parallel_size,
#         max_model_len=2048,
#         max_num_batched_tokens=4096,
#         max_num_seqs=16
#     )

#     curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

#     results = []
#     # for jsonl
#     # with open(args.dataset_path, "r", encoding="utf-8") as f:
#     #     lines = f.readlines()
#     with open(args.dataset_path, "r", encoding="utf-8") as f:
#         samples = json.load(f)

#     # for line in tqdm(lines, desc="Processing", unit="sample"):
#     #     sample = json.loads(line.strip())
#     for sample in tqdm(samples, desc="Processing", unit="sample"):    
#         q = sample["question"]
#         golden = sample.get("golden_answers", [])

#         prediction, turns = run_reasoning(q, llm, curr_search_template, args.max_turns)

#         results.append({
#             "question": q,
#             "golden_answers": golden,
#             "prediction": prediction,
#             "turns": turns
#         })

#     with open(args.output_file, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     print(f"✅ Finished. Results saved to {args.output_file}")


# if __name__ == "__main__":
#     main()

import json
import re
import argparse
import requests
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch


def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def search(query: str, port: int):
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    response = requests.post(
        f"http://127.0.0.1:{port}/retrieve",
        json=payload
    )
    results = response.json()["result"]

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


def run_reasoning(question, llm, curr_search_template, max_turns, retriever_port):
    if question[-1] != "?":
        question += "?"

    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    cnt = 0
    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n",
        "</search>\n\n", " </search>\n\n"
    ]

    while True:
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
            stop=target_sequences,
        )

        outputs = llm.generate([prompt], sampling_params)
        output_text = outputs[0].outputs[0].text

        if not any(s in output_text for s in target_sequences):
            output_text += target_sequences[0]

        # 如果已经给出最终答案
        if "</answer>" in output_text:
            prompt += output_text
            break

        tmp_query = get_query(output_text)
        if tmp_query:
            search_results = search(tmp_query, retriever_port)
        else:
            search_results = ""

        search_text = curr_search_template.format(
            output_text=output_text,
            search_results=search_results
        )
        prompt += search_text

        cnt += 1
        if cnt >= max_turns:
            prompt += "\n<answer> Unable to complete reasoning within max turns. </answer>\n"
            break

    return prompt, cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model ID for vLLM model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to input dataset (json or jsonl)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save predictions (json)")
    parser.add_argument("--max_turns", type=int, default=5, help="Maximum number of reasoning-search turns")
    parser.add_argument(
        "--retriever_port",
        type=int,
        default=8205,
        help="Port of the retrieval server"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs for tensor parallelism"
    )
    args = parser.parse_args()

    llm = LLM(
        model=args.model_id,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        max_num_seqs=16
    )

    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    results = []

    # 目前按 json list 读取（与你当前代码一致）
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for sample in tqdm(samples, desc="Processing", unit="sample"):
        q = sample["question"]
        golden = sample.get("golden_answers", [])

        prediction, turns = run_reasoning(
            q,
            llm,
            curr_search_template,
            args.max_turns,
            args.retriever_port
        )

        results.append({
            "question": q,
            "golden_answers": golden,
            "prediction": prediction,
            "turns": turns
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Finished. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
