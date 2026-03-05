import argparse
import json
from openai import OpenAI

from agent.tool.tool_env import ToolEnv, step_batch
from agent.tool.tools import _default_tools
import re
import copy
import pdb
import os
from tqdm import tqdm

# ANSI color codes for colored output
COLORS = {
    "user": "\033[1;34m",      # Bold Blue
    "assistant": "\033[1;32m",  # Bold Green
    "tool": "\033[1;33m",       # Bold Yellow
    "tool_call": "\033[1;35m",  # Bold Purple
    "reset": "\033[0m",         # Reset to default
    "bg_user": "\033[44m",      # Blue background
    "bg_assistant": "\033[42m", # Green background
    "bg_tool": "\033[43m",      # Yellow background
    "bg_tool_call": "\033[45m", # Purple background
}

def parse_args():
    parser = argparse.ArgumentParser(description='Run VLLM inference with configurable parameters')
    parser.add_argument('--api-key', type=str, default="EMPTY",
                        help='OpenAI API key')
    parser.add_argument('--api-base', type=str, default="http://localhost:8002/v1",
                        help='OpenAI API base URL')
    parser.add_argument('--model', type=str, default="agent",
                        help='Model name for inference')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling')
    parser.add_argument('--top-p', type=float, default=1.0,
                        help='Top-p for nucleus sampling')
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--max-turns', type=int, default=3,
                        help='Maximum turns of search')
    parser.add_argument('--question', type=str, default="Which magazine came out first, Tit-Bits or Illustreret Nyhedsblad?",
                        help='Question to ask the model')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--dataset', type=str, default="HotpotQA", help='infer dataset')
    parser.add_argument('--output_dir', type=str, default="HotpotQA", help='infer dataset')
    parser.add_argument('--port_config', type=str, default="", help='API service port')
    return parser.parse_args()

def process_tool_call(responses_str):

    def process_single_response(resp):
        eos_token = "<|im_end|>"
        tool_call_end ="</query>"
        tool_pattern = r'<query>(.*?)</query>'
        match = re.search(tool_pattern, resp, re.DOTALL)
        
        if not match:
            return resp + eos_token, False  # No tool call found
        
        resp = resp.split(tool_call_end)[0] + tool_call_end
        
        return resp + eos_token, True
    
    # Process each response string
    return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

def execute_tool_calls_batch(response_strs, env, active_masks, data_sources):
    tool_custom_response_template = "<|im_start|>user\n<knowledge>\n{tool_response}\n</knowledge><|im_end|>\n<|im_start|>assistant\n<think>"
    active_envs = []
    active_responses = []
    active_indices = []
    
    for i, (resp, active) in enumerate(zip(response_strs, active_masks)):
        if active:
            active_envs.append(env)
            active_responses.append(resp)
            active_indices.append(i)
    
    # Initialize result list with empty strings
    tool_responses = [""] * len(response_strs)
    
    if not active_envs:
        return tool_responses
        
    # Use the independent step_batch function for active environments
    batch_results = step_batch(active_envs, active_responses, data_sources)
    
    # Map results back to original indices
    for idx, result in zip(active_indices, batch_results):
        if result is None:
            tool_responses[idx] = ""
        else:
            tool_response = result[0]
            tool_responses[idx] = tool_custom_response_template.format(tool_response=tool_response)
    return tool_responses

def colorprint(mode, r_str, t_str, use_colors):
    if not r_str.startswith("<think>\n"):
        r_str = "<think>\n" + r_str
    
    if mode is True:
        think = re.findall(r'<think>(.*?)</think>', r_str, re.DOTALL)[0]
        if not think.endswith("\n"):
            think += "\n"   
        # query = re.findall(r'<query>\n{"query": "(.*?)"\n}\n</query>', r_str, re.DOTALL)[0]
        query = re.search(r'<query>\s*{"query":\s*"(.*?)"\s*}\s*</query>', r_str, re.DOTALL)[0]
        knowledge = re.findall(r'<knowledge>(.*?)</knowledge>', t_str, re.DOTALL)[0]
        knowledge_list = json.loads(knowledge)['results']
        knowledge = "\n"
        for k in knowledge_list:
            knowledge += str(k) + "\n"
        
        if use_colors:
            print(f"\n{COLORS['bg_tool_call']} Think {COLORS['reset']} {COLORS['tool_call']}{think}{COLORS['reset']}")
            print(f"{COLORS['tool_call']}Query:{COLORS['reset']}\n{query}{COLORS['reset']}")
            print(f"\n{COLORS['bg_tool']} Knowledge {COLORS['reset']} {COLORS['tool']}{knowledge}{COLORS['reset']}")
        else:
            print(f"\n[Think] {think}")
            print(f"Query:\n{query}")
            print(f"\nKnowledge: {knowledge}") 
    else:
        think = re.findall(r'<think>(.*?)</think>', r_str, re.DOTALL)[0]
        if not think.endswith("\n"):
            think += "\n"

        answer = re.findall(r'<answer>\n(.*?)\n</answer>', r_str, re.DOTALL)[0]
        
        if use_colors:
            print(f"\n{COLORS['bg_tool_call']} Think {COLORS['reset']} {COLORS['tool_call']}{think}{COLORS['reset']}")
            print(f"{COLORS['tool_call']}Answer:{COLORS['reset']}\n{answer}{COLORS['reset']}")
        else:
            print(f"\n[Think] {think}")
            print(f"Answer:\n{answer}")
            
        print("\n")

def main():
    args = parse_args()
    use_colors = not args.no_color
    OPENAI_API_KEY = args.api_key
    OPENAI_API_BASE = args.api_base
    MODEL_NAME = args.model
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    MAX_TURNS = args.max_turns
    DATASET = args.dataset
    OUTPUT_DIR = args.output_dir
    PORT_CONFIG = args.port_config
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    # Set up tools
    tools = _default_tools("search", tool_config_path=PORT_CONFIG)
    env = ToolEnv(tools=tools, max_turns=MAX_TURNS)

    qa_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets", DATASET, "raw", "qa_test_full.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
    # Test only first 1000 samples
    qa_list = qa_list[:1000]
    results = []

    out_path = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 检查已推理的数据点
    processed_questions = set()
    if os.path.exists(out_path):
        print(f"检测到已有输出文件：{out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        qa_out = json.loads(line)
                        if "question" in qa_out:
                            processed_questions.add(qa_out["question"])
                    except json.JSONDecodeError:
                        continue
        print(f"⚠️ 已推理 {len(processed_questions)} 条，将跳过已推理的数据点。")
    else:
        print("没有已有的输出文件，将从头开始推理。")
    
    out_file = open(out_path, "a", encoding="utf-8")

    for idx, qa in enumerate(tqdm(qa_list, desc="Running QA", ncols=100)):
        # 跳过已推理的数据点
        if qa["question"] in processed_questions:
            tqdm.write(f"[{idx+1}/{len(qa_list)}] skipped (already processed).")
            continue
        question_raw = qa["question"]
        messages = [{
            "role": "user",
            "content": '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"name": "search", "description": "Search for information on the internet using Wikipedia as a knowledge source.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nAnswer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.\nYou must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.\nWhen you have the final answer, you can output the answer inside <answer>...</answer>.\n\nOutput format for tool call:\n<think>\n...\n</think>\n<query>\n...\n</query>\n\nOutput format for answer:\n<think>\n...\n</think>\n<answer>\n...</answer>\nQuestion: '+question_raw+'<|im_end|>\n<|im_start|>assistant\n'
        }]
        prompt_final = None
        turns = 0
        for step in range(MAX_TURNS):
            turns = step + 1
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_tokens=MAX_TOKENS,
                )
                response_message = response.choices[0].message
                responses_str = [response_message.content]
                responses_str, active_masks = process_tool_call(responses_str)
                tool_responses = execute_tool_calls_batch(responses_str, env, active_masks, [DATASET]*len(responses_str))
                if active_masks[0] is True:
                    prompt = messages[0]["content"]+responses_str[0]+tool_responses[0]
                    messages = [{
                        "role": "user",
                        "content": prompt
                    }]
                else:
                    prompt_final = messages[0]["content"]+responses_str[0]
                    break
            except Exception as e:
                continue
        # 记录结果
        qa_out = dict(qa)
        qa_out["prediction"] = prompt_final if prompt_final is not None else ""
        qa_out["turns"] = turns

        # 每条写入一行 JSON（json lines 格式）
        out_file.write(json.dumps(qa_out, ensure_ascii=False) + "\n")
        out_file.flush()  # 立刻写入磁盘，防崩溃

        tqdm.write(f"[{idx+1}/{len(qa_list)}] done.")
    out_file.close()
    print(f"All predictions written to {out_path}")

if __name__ == "__main__":
    main() 

# import argparse
# import json
# from openai import OpenAI

# from agent.tool.tool_env import ToolEnv, step_batch
# from agent.tool.tools import _default_tools
# import re
# import copy
# import pdb
# import os
# from tqdm import tqdm


# # ANSI color codes for colored output
# COLORS = {
#     "user": "\033[1;34m",      # Bold Blue
#     "assistant": "\033[1;32m",  # Bold Green
#     "tool": "\033[1;33m",       # Bold Yellow
#     "tool_call": "\033[1;35m",  # Bold Purple
#     "reset": "\033[0m",         # Reset to default
#     "bg_user": "\033[44m",      # Blue background
#     "bg_assistant": "\033[42m", # Green background
#     "bg_tool": "\033[43m",      # Yellow background
#     "bg_tool_call": "\033[45m", # Purple background
# }


# def parse_args():
#     parser = argparse.ArgumentParser(description='Run VLLM inference with configurable parameters')
#     parser.add_argument('--api-key', type=str, default="EMPTY",
#                         help='OpenAI API key')
#     parser.add_argument('--api-base', type=str, default="http://localhost:8002/v1",
#                         help='OpenAI API base URL')
#     parser.add_argument('--model', type=str, default="agent",
#                         help='Model name for inference')
#     parser.add_argument('--temperature', type=float, default=1.0,
#                         help='Temperature for sampling')
#     parser.add_argument('--top-p', type=float, default=1.0,
#                         help='Top-p for nucleus sampling')
#     parser.add_argument('--max-tokens', type=int, default=4096,
#                         help='Maximum number of tokens to generate')
#     parser.add_argument('--max-turns', type=int, default=20,
#                         help='Maximum turns of search')
#     parser.add_argument('--question', type=str,
#                         default="Which magazine came out first, Tit-Bits or Illustreret Nyhedsblad?",
#                         help='Question to ask the model')
#     parser.add_argument('--no-color', action='store_true',
#                         help='Disable colored output')
#     parser.add_argument('--dataset', type=str, default="HotpotQA", help='infer dataset')
#     parser.add_argument('--output_dir', type=str, default="HotpotQA_output.jsonl", help='output file path')
#     parser.add_argument('--port_config', type=str, default="", help='API service port')
#     return parser.parse_args()


# def process_tool_call(responses_str):

#     def process_single_response(resp):
#         eos_token = "<|im_end|>"
#         tool_call_end = "</query>"
#         tool_pattern = r'<query>(.*?)</query>'
#         match = re.search(tool_pattern, resp, re.DOTALL)

#         if not match:
#             return resp + eos_token, False  # No tool call found

#         resp = resp.split(tool_call_end)[0] + tool_call_end
#         return resp + eos_token, True

#     return [process_single_response(resp)[0] for resp in responses_str], \
#            [process_single_response(resp)[1] for resp in responses_str]


# def execute_tool_calls_batch(response_strs, env, active_masks, data_sources):
#     tool_custom_response_template = "<|im_start|>user\n<knowledge>\n{tool_response}\n</knowledge><|im_end|>\n<|im_start|>assistant\n<think>"
#     active_envs = []
#     active_responses = []
#     active_indices = []
    
#     for i, (resp, active) in enumerate(zip(response_strs, active_masks)):
#         if active:
#             active_envs.append(env)
#             active_responses.append(resp)
#             active_indices.append(i)
    
#     # Initialize result list with empty strings
#     tool_responses = [""] * len(response_strs)
    
#     if not active_envs:
#         return tool_responses
        
#     # Use the independent step_batch function for active environments
#     batch_results = step_batch(active_envs, active_responses, data_sources)
    
#     # Map results back to original indices
#     for idx, result in zip(active_indices, batch_results):
#         if result is None:
#             tool_responses[idx] = ""
#         else:
#             tool_response = result[0]
#             tool_responses[idx] = tool_custom_response_template.format(tool_response=tool_response)
#     return tool_responses


# def main():
#     args = parse_args()
#     use_colors = not args.no_color
#     OPENAI_API_KEY = args.api_key
#     OPENAI_API_BASE = args.api_base
#     MODEL_NAME = args.model
#     TEMPERATURE = args.temperature
#     TOP_P = args.top_p
#     MAX_TOKENS = args.max_tokens
#     MAX_TURNS = args.max_turns
#     DATASET = args.dataset
#     OUTPUT_FILE = args.output_dir
#     PORT_CONFIG = args.port_config

#     # Initialize OpenAI client
#     client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

#     # Set up tool environment
#     tools = _default_tools("search", tool_config_path=PORT_CONFIG)
#     env = ToolEnv(tools=tools, max_turns=MAX_TURNS)

#     # Load dataset
#     qa_path = os.path.join(
#         os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
#         "datasets", DATASET, "raw", "qa_test_full.json"
#     )
#     with open(qa_path, "r", encoding="utf-8") as f:
#         qa_list = json.load(f)

#     # ---------------------------
#     # 🔥 Resume function
#     # ---------------------------
#     existed = 0
#     if os.path.exists(OUTPUT_FILE):
#         print(f"检测到已有输出文件：{OUTPUT_FILE}")
#         with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
#             for line in f:
#                 if line.strip():
#                     existed += 1
#         print(f"⚠️ 已推理 {existed} 条，将从第 {existed + 1} 条继续推理。")
#     else:
#         print("没有已有的输出文件，将从头开始推理。")

#     out_file = open(OUTPUT_FILE, "a", encoding="utf-8")

#     # ---------------------------
#     # Main inference loop
#     # ---------------------------
#     for idx in range(existed, len(qa_list)):
#         qa = qa_list[idx]
#         question_raw = qa["question"]

#         messages = [{
#             "role": "user",
#             "content": '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"name": "search", "description": "Search for information on the internet using Wikipedia as a knowledge source.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nAnswer the given question. You can query from knowledge base provided to you to answer the question. You can query knowledge as many times as you want.\nYou must first conduct reasoning inside <think>...</think>. If you need to query knowledge, you can set a query statement between <query>...</query> to query from knowledge base after <think>...</think>.\nWhen you have the final answer, you can output the answer inside <answer>...</answer>.\n\nOutput format for tool call:\n<think>\n...\n</think>\n<query>\n...\n</query>\n\nOutput format for answer:\n<think>\n...\n</think>\n<answer>\n...</answer>\nQuestion: '+question_raw+'<|im_end|>\n<|im_start|>assistant\n'
#         }]

#         prompt_final = None
#         turns = 0

#         # Multi-turn reasoning
#         for step in range(MAX_TURNS):
#             turns = step + 1
#             try:
#                 response = client.chat.completions.create(
#                     model=MODEL_NAME,
#                     messages=messages,
#                     temperature=TEMPERATURE,
#                     top_p=TOP_P,
#                     max_tokens=MAX_TOKENS,
#                 )

#                 response_message = response.choices[0].message
#                 responses_str = [response_message.content]

#                 responses_str, active_masks = process_tool_call(responses_str)
#                 tool_responses = execute_tool_calls_batch(
#                     responses_str, env, active_masks, [DATASET]*len(responses_str)
#                 )

#                 # Tool call case
#                 if active_masks[0]:
#                     prompt = messages[0]["content"] + responses_str[0] + tool_responses[0]
#                     messages = [{"role": "user", "content": prompt}]
#                 else:
#                     prompt_final = messages[0]["content"] + responses_str[0]
#                     break

#             except Exception as e:
#                 print("Error:", e)
#                 continue

#         # Save result
#         qa_out = dict(qa)
#         qa_out["prediction"] = prompt_final if prompt_final else ""
#         qa_out["turns"] = turns

#         out_file.write(json.dumps(qa_out, ensure_ascii=False) + "\n")
#         out_file.flush()

#         tqdm.write(f"[{idx+1}/{len(qa_list)}] Done.")

#     out_file.close()
#     print(f"所有结果已写入：{OUTPUT_FILE}")


# if __name__ == "__main__":
#     main()
