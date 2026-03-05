# run_rag_agent.py
import os
import json
import time
import re
import requests
from tqdm import tqdm
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
import argparse

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from bing_search import bing_web_search, extract_relevant_info, fetch_page_content
from evaluate import run_evaluation
from prompts import (
    get_singleqa_rag_agent_instruction, 
    get_multiqa_rag_agent_instruction, 
    get_gpqa_rag_agent_instruction, 
    get_math_rag_agent_instruction, 
    get_code_rag_agent_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

# Define special symbols
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_URL = "<|begin_url|>"
END_URL = "<|end_url|>"
BEGIN_FULL_PAGE = "<|begin_full_page|>"
END_FULL_PAGE = "<|end_full_page|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG Agent for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'nq', 'triviaqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'medmcqa', 'pubhealth'],
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['test', 'diamond', 'main', 'extended'],
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # RAG Agent configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=5,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_url_fetch',
        type=int,
        default=5,
        help="Maximum number of URL fetches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=10,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    # Model configuration
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=True,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        required=True,
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Extract arguments
    dataset_name = args.dataset_name
    split = args.split
    subset_num = args.subset_num
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_URL_FETCH = args.max_url_fetch
    MAX_TURN = args.max_turn
    top_k = args.top_k
    model_path = args.model_path
    temperature = args.temperature
    top_p = args.top_p
    top_k_sampling = args.top_k_sampling
    repetition_penalty = args.repetition_penalty
    max_tokens = args.max_tokens
    bing_subscription_key = args.bing_subscription_key
    bing_endpoint = args.bing_endpoint
    use_jina = args.use_jina
    jina_api_key = args.jina_api_key

    # Adjust parameters based on dataset
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        MAX_SEARCH_LIMIT = 5
        MAX_URL_FETCH = 5
        MAX_TURN = 10
        top_k = 5

    # Set default repetition_penalty if not provided
    if repetition_penalty is None:
        repetition_penalty = 1.05 if 'qwq' in model_path.lower() else 1.0
    
    if args.jina_api_key == 'None':
        jina_api_key = None

    # Data paths based on dataset
    if dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Define function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Define output directory based on model and dataset
    if 'qwq' in model_path.lower():
        if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.rag_agent'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.rag_agent'
    else:
        model_short_name = model_path.split('/')[-1].lower().replace('-instruct', '')
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.rag_agent'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the LLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
    )

    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)

    # ---------------------- Prepare Input ----------------------
    input_list = []
    for item in filtered_data:
        question = item['Question']
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            if dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH)
            elif dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
                instruction = get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif dataset_name in ['math500', 'aime', 'amc']:
            instruction = get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)

        elif dataset_name == 'gpqa':
            instruction = get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH)
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'llama' in model_path.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif dataset_name == 'livecode':
            instruction = get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH)
            question_title = item.get('question_title', '')
            if 'qwq' in model_path.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    if subset_num != -1:
        input_list = input_list[:subset_num]
        filtered_data = filtered_data[:subset_num]

    # Initialize active sequences with search and URL fetch counters
    active_sequences = [{
        'item': item,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'pending_operations': [],  # Queue of operations to execute
        'executed_search_queries': set(),
        'executed_url_fetches': set(),
        'search_count': 0  # Search counter
    } for item, prompt in zip(filtered_data, input_list)]

    # ---------------------- Set Max Tokens ----------------------
    if 'qwq' in model_path.lower():
        if dataset_name in ['aime', 'amc', 'livecode']:
            max_tokens = 32768
        else:
            max_tokens = 20480
    else:
        max_tokens = 8192

    # ---------------------- Generation Function ----------------------
    def run_generation(sequences, max_tokens):
        """
        Run LLM generation on provided sequences.
        """
        prompts = [s['prompt'] for s in sequences]
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k_sampling, 
            repetition_penalty=repetition_penalty,
            stop=[END_SEARCH_QUERY, END_URL, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        output_list = llm.generate(prompts, sampling_params=sampling_params)
        return output_list

    # Function to extract text between two tags
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    start_time = time.time()
    turn = 0

    # Main loop until all sequences are completed
    while True:
        # Separate sequences with pending operations and those needing generation
        sequences_with_pending_ops = [seq for seq in active_sequences if not seq['finished'] and seq['pending_operations']]
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished'] and not seq['pending_operations']]

        # First, handle pending operations
        if sequences_with_pending_ops:
            print(f"{len(sequences_with_pending_ops)} sequences have pending operations. Executing...")
            for seq in sequences_with_pending_ops:
                # Execute the next pending operation
                operation = seq['pending_operations'].pop(0)  # FIFO
                op_type = operation['type']
                content = operation['content']

                if op_type == 'search':
                    query = content
                    if query in search_cache:
                        results = search_cache[query]
                        print(f"Using cached search results for query: {query}")
                    else:
                        try:
                            # Execute search and cache results
                            results = bing_web_search(query, bing_subscription_key, bing_endpoint, market='en-US', language='en')
                            search_cache[query] = results
                            print(f"Executed and cached search for query: {query}")
                        except Exception as e:
                            print(f"Error during search query '{query}': {e}")
                            search_cache[query] = {}
                            results = {}
                    relevant_info = extract_relevant_info(results)[:top_k]
                    search_result_str = json.dumps(relevant_info, ensure_ascii=False, indent=2)
                    # Append search results to the prompt
                    append_text = f"\n{BEGIN_SEARCH_RESULT}\n{search_result_str}\n{END_SEARCH_RESULT}\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    # Update history
                    seq['history'].append(append_text)
                    # Increment search count
                    seq['search_count'] += 1

                elif op_type == 'fetch_url':
                    urls = content
                    # Calculate remaining URL fetches
                    remaining_fetches = MAX_URL_FETCH - len(seq['executed_url_fetches'])
                    if remaining_fetches <= 0:
                        # Reached URL fetch limit, add limit message and mark sequence as finished
                        limit_message = f"\n{BEGIN_FULL_PAGE}\nThe maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n{END_FULL_PAGE}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print("Reached URL fetch limit. Sequence marked as finished.")
                        continue

                    # Split and clean URLs
                    urls_to_fetch = [u.strip() for u in urls.split(",")]
                    # Filter already fetched URLs
                    urls_to_fetch = [u for u in urls_to_fetch if u not in seq['executed_url_fetches']]
                    # Limit the number of URLs to fetch
                    urls_to_fetch = urls_to_fetch[:remaining_fetches]

                    if not urls_to_fetch:
                        print("All requested URLs have been fetched or exceeded the limit.")
                        continue

                    # Batch fetch page content, considering cache
                    urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                    cached_urls = [u for u in urls_to_fetch if u in url_cache]
                    
                    fetched_contents = []

                    # Use cached URL content
                    for url in cached_urls:
                        content = url_cache[url]
                        print(f"Using cached URL content for URL: {url}")
                        fetched_contents.append((url, content))
                    
                    # Batch fetch uncached URLs
                    if urls_to_fetch_filtered:
                        try:
                            # Batch pass uncached URLs
                            contents = fetch_page_content(urls_to_fetch_filtered, use_jina=use_jina, jina_api_key=jina_api_key)
                            for url, content in contents.items():
                                url_cache[url] = content
                                print(f"Fetched and cached URL content for URL: {url}")
                                fetched_contents.append((url, content))
                        except Exception as e:
                            for url in urls_to_fetch_filtered:
                                content = f"Error fetching URL: {e}"
                                url_cache[url] = content
                                fetched_contents.append((url, content))
                                print(f"Error fetching URL '{url}': {e}")

                    # Update fetched URLs
                    for url, _ in fetched_contents:
                        seq['executed_url_fetches'].add(url)

                    # Construct full page content string
                    fetched_pages = dict(fetched_contents)
                    full_page_str = json.dumps(fetched_pages, ensure_ascii=False, indent=2)
                    # Append full page content to the prompt
                    append_text = f"\n{BEGIN_FULL_PAGE}\n{full_page_str}\n{END_FULL_PAGE}\n"
                    seq['prompt'] += append_text
                    seq['output'] += append_text
                    # Update history
                    seq['history'].append(append_text)

                    print(f"Fetched and cached {len(fetched_contents)} URLs.")

                    # Check if URL fetch limit is reached
                    if len(seq['executed_url_fetches']) >= MAX_URL_FETCH:
                        limit_message = f"\n{BEGIN_FULL_PAGE}\nThe maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n{END_FULL_PAGE}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print("Reached URL fetch limit. Sequence marked as finished.")

        # Continue to the next iteration if there are pending operations
        if sequences_with_pending_ops:
            continue  # Process operations first

        # Handle sequences needing generation
        if sequences_needing_generation:
            turn += 1
            print(f"Turn {turn}: {len(sequences_needing_generation)} sequences need generation. Generating with LLM...")
            outputs = run_generation(sequences_needing_generation, max_tokens)
            print("Generation complete. Processing outputs...")

            # Process each generated output
            for seq, out in zip(sequences_needing_generation, outputs):
                text = out.outputs[0].text
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text

                # Check if the generated content contains search queries or URL fetch requests
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                url_fetch = extract_between(text, BEGIN_URL, END_URL)

                if search_query:
                    # Check if search limit is not exceeded
                    if seq['search_count'] < MAX_SEARCH_LIMIT:
                        # Check if this search query has not been executed
                        if search_query not in seq['executed_search_queries']:
                            # Add search operation to pending queue
                            seq['pending_operations'].append({'type': 'search', 'content': search_query})
                            seq['executed_search_queries'].add(search_query)
                            print(f"Added pending search operation for query: {search_query}")
                        else:
                            print(f"Search query already executed: {search_query}")
                    else:
                        # Add limit message if search limit is exceeded
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit exceeded for query: {search_query}")

                if url_fetch:
                    # Check if URL fetch limit is not exceeded
                    if len(seq['executed_url_fetches']) < MAX_URL_FETCH:
                        # Split and check if URLs have already been fetched
                        urls = [u.strip() for u in url_fetch.split(",")]
                        urls_to_fetch = [u for u in urls if u not in seq['executed_url_fetches']]
                        if urls_to_fetch:
                            # Add URL fetch operation to pending queue
                            seq['pending_operations'].append({'type': 'fetch_url', 'content': ', '.join(urls_to_fetch)})
                            print(f"Added pending URL fetch operation for URLs: {urls_to_fetch}")
                        else:
                            print(f"All requested URLs have been fetched or exceeded the limit: {urls}")
                    else:
                        # Add limit message if URL fetch limit is exceeded
                        limit_message = f"\n{BEGIN_FULL_PAGE}\nThe maximum number of URL fetches has been reached. You are not allowed to fetch more URLs.\n{END_FULL_PAGE}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print("URL fetch limit exceeded.")

                # If no new operations are added, mark sequence as finished
                if not search_query and not url_fetch:
                    seq['finished'] = True
                    print("Sequence marked as finished.")

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= MAX_TURN:
                print(f"Exceeded maximum number of turns ({MAX_TURN}). Stopping.")
                break
            # Optionally, implement a delay or other logic to prevent infinite loops
            pass

    total_time = time.time() - start_time

    # Collect all outputs
    output_list = [seq['output'] for seq in active_sequences]

    # ---------------------- Evaluation ----------------------
    print("Evaluating generated answers...")
    run_evaluation(
        filtered_data=filtered_data,
        input_list=input_list,
        output_list=output_list,
        dataset_name=dataset_name,
        output_dir=output_dir,
        total_time=total_time,
        split=split,
    )

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

if __name__ == "__main__":
    main()
