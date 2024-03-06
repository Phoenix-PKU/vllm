import time, sys, os, json, argparse
from typing import List, Tuple
from tqdm import tqdm
import torch
import numpy as np

from typing import Optional
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm import LLM

path = '/data/leili/datasets/ShareGPT52K/sg_90k_part1.json'
max_length = 2048
num_requests = 500
max_output_tokens = 256
sampling_params = SamplingParams(max_tokens=max_output_tokens)

def create_test_prompts() -> List[str]:
    prompts = []
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for entry in data:
        for topic in entry['conversations']:
            if topic['from'] == 'human':
                if len(topic['value']) < max_length and len(topic['value']) != 0:
                    prompts.append(topic['value'])

    return prompts[:num_requests]

def run_to_completion(engine: LLMEngine):
    start_time = time.perf_counter()
    
    num_iters = 0
    latency_step_list = []
    block_used_list = []

    tps = engine.parallel_config.tensor_parallel_size
    total_blocks = engine.cache_config.num_gpu_blocks
    block_size = engine.cache_config.block_size
    max_num_seqs = engine.scheduler_config.max_num_seqs

    # TODO(leili): add tqdm.
    while engine.has_unfinished_requests():
        free_blocks = engine.scheduler.block_manager.get_num_free_gpu_blocks()
        
        before_step = time.perf_counter()
        request_outputs = engine.step()
        after_step = time.perf_counter()

        if len(engine.metadata_list) == max_num_seqs:
            latency_step_list.append(after_step - before_step)

        block_used_list.append(total_blocks - free_blocks)
        num_iters += 1
        """
        for output in request_outputs:
            if output.finished:
                print("block used: ", total_blocks - free_blocks)
        """

        """
        print(len(engine.metadata_list), \
            engine.get_num_unfinished_requests(), \
            engine.cache_config.num_gpu_blocks, \
            engine.scheduler.block_manager.get_num_free_gpu_blocks(), \
        )
        """

    end_time = time.perf_counter()
    latency = end_time - start_time
    print("model:", engine.model_config.model)
    print("num requests:", num_requests, "max output tokens:", max_output_tokens)
    print("total latency(s):", latency, "num_iters:", num_iters) 
    print("avg latency(s, seq full):", sum(latency_step_list)/len(latency_step_list))
    print("num gpu blocks:", total_blocks, ", block size(*?KiB):", block_size)
    print("max blocks used:", max(block_used_list), "batch size(max num seqs):", max_num_seqs)
    return latency

def add_requests(engine: LLMEngine,
                     test_prompts: List[str]):
    nums = len(test_prompts)
    for request_id in range(nums):
        prompt = test_prompts[request_id]
        engine.add_request(str(request_id), prompt, sampling_params)

def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    test_prompts = create_test_prompts()
    # process_requests(engine, test_prompts)
    add_requests(engine, test_prompts)
    latency = run_to_completion(engine)
    """
    num_iters = 3
    latencies = []
    for _ in tqdm(range(num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(engine))
    print(f'Avg latency: {np.mean(latencies)} seconds')
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)