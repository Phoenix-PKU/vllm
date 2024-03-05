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
max_test_num = 1
sampling_params = SamplingParams(max_tokens=256)

def create_test_prompts() -> List[str]:
    return ["Hello, my name is",]

def run_to_completion(engine: LLMEngine,
                    profile_dir: Optional[str] = None):
    if profile_dir:
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
                profile_memory=True) as p:
            engine.step()
        print(p.key_averages())
    else:
        start_time = time.perf_counter()
        engine.step()
        end_time = time.perf_counter()
        latency = end_time - start_time
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
    print("Warming up...")
    run_to_completion(engine, profile_dir=None)
    result_dir = "./benchmark_latency/"
    run_to_completion(engine, profile_dir=result_dir)
    num_iters = 3
    latencies = []
    for _ in tqdm(range(num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(engine, profile_dir=None))
    print(f'Avg latency: {np.mean(latencies)} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)