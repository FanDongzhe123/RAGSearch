import argparse
import json
from openai import OpenAI

from agent.tool.tool_env import ToolEnv, step_batch
from agent.tool.tools import _default_tools
import re
import copy
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Run VLLM inference with configurable parameters')
    parser.add_argument('--api-key', type=str, default="EMPTY")
    parser.add_argument('--api-base', type=str, default="http://localhost:8002/v1")
    parser.add_argument('--model', type=str, default="agent")
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--max-turns', type=int, default=5)
    parser.add_argument('--question', type=str,
                        default="Which magazine came out first, Tit-Bits or Illustreret Nyhedsblad?")
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--dataset', type=str, default="HotpotQA")
    parser.add_argument('--output_dir', type=str, default="HotpotQA")
    parser.add_argument('--port_config', type=str, default="")
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


def process_tool_call(responses_str):

    def process_single_response(resp):
        eos_token = "<|im_end|>"
        tool_call_end = "</query>"
        tool_pattern = r'<query>(.*?)</query>'
        match = re.search(tool_pattern, resp, re.DOTALL)

        if not match:
            return resp + eos_token, False

        resp = resp.split(tool_call_end)[0] + tool_call_end
        return resp + eos_token, True

    new_resps, masks = [], []
    for r in responses_str:
        r2, m = process_single_response(r)
        new_resps.append(r2)
        masks.append(m)
    return new_resps, masks


def execute_tool_calls_batch(response_strs, envs, active_masks, data_sources):
    tool_custom_response_template = (
        "<|im_start|>user\n<knowledge>\n{tool_response}\n</knowledge><|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )

    active_envs, active_responses, active_indices = [], [], []

    for i, (resp, active) in enumerate(zip(response_strs, active_masks)):
        if active:
            active_envs.append(envs[i])
            active_responses.append(resp)
            active_indices.append(i)

    tool_responses = [""] * len(response_strs)

    if not active_envs:
        return tool_responses

    batch_results = step_batch(active_envs, active_responses, data_sources)

    for idx, result in zip(active_indices, batch_results):
        if result is not None:
            tool_responses[idx] = tool_custom_response_template.format(
                tool_response=result[0]
            )

    return tool_responses


def main():
    args = parse_args()

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    tools = _default_tools("search", tool_config_path=args.port_config)

    qa_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "datasets", args.dataset, "raw", "qa_test_full.json"
    )
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    out_path = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_file = open(out_path, "a", encoding="utf-8")

    # =============================
    # batch over samples（保留）
    # =============================
    for b in tqdm(range(0, len(qa_list), args.batch_size), desc="Running QA", ncols=100):
        qa_batch = qa_list[b: b + args.batch_size]
        batch_size = len(qa_batch)

        messages_batch = []
        envs = []

        for qa in qa_batch:
            question_raw = qa["question"]
            messages = [{
                "role": "user",
                "content":
                    '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n'
                    '# Tools\n\n'
                    'You may call one or more functions to assist with the user query.\n\n'
                    'You are provided with function signatures within <tools></tools> XML tags:\n'
                    '<tools>\n'
                    '{"name": "search", "description": "Search for information on the internet using Wikipedia as a knowledge source.", '
                    '"parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, '
                    '"required": ["query"]}}\n'
                    '</tools>\n\n'
                    '<|im_end|>\n'
                    '<|im_start|>user\n'
                    'Answer the given question.\n'
                    'You must first conduct reasoning inside <think>...</think>.\n'
                    'If you need to query knowledge, you can set a query statement between <query>...</query>.\n'
                    'When you have the final answer, output it inside <answer>...</answer>.\n\n'
                    f'Question: {question_raw}'
                    '<|im_end|>\n<|im_start|>assistant\n'
            }]
            messages_batch.append(messages)
            envs.append(ToolEnv(tools=tools, max_turns=args.max_turns))

        finished = [False] * batch_size
        prompt_final = [""] * batch_size
        turns = [0] * batch_size

        # =============================
        # multi-turn loop
        # =============================
        for step in range(args.max_turns):

            responses_str = [""] * batch_size
            active_masks = [False] * batch_size

            # -------- LLM 串行（不 batch）--------
            for i in range(batch_size):
                if finished[i]:
                    continue

                turns[i] += 1

                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages_batch[i],   # ✅ 单条
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                )

                resp = response.choices[0].message.content
                resp, mask = process_tool_call([resp])
                responses_str[i] = resp[0]
                active_masks[i] = mask[0]

            # -------- Retriever batch（你要的）--------
            tool_responses = execute_tool_calls_batch(
                responses_str,
                envs,
                active_masks,
                [args.dataset] * batch_size
            )

            # -------- 写回 --------
            for i in range(batch_size):
                if finished[i]:
                    continue

                if active_masks[i]:
                    messages_batch[i] = [{
                        "role": "user",
                        "content":
                            messages_batch[i][0]["content"]
                            + responses_str[i]
                            + tool_responses[i]
                    }]
                else:
                    prompt_final[i] = (
                        messages_batch[i][0]["content"]
                        + responses_str[i]
                    )
                    finished[i] = True

            if all(finished):
                break

        # =============================
        # write results
        # =============================
        for i, qa in enumerate(qa_batch):
            qa_out = dict(qa)
            qa_out["prediction"] = prompt_final[i]
            qa_out["turns"] = turns[i]
            out_file.write(json.dumps(qa_out, ensure_ascii=False) + "\n")
            out_file.flush()

    out_file.close()
    print(f"All predictions written to {out_path}")


if __name__ == "__main__":
    main()
