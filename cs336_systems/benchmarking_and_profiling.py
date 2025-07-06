import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl
from cs336_basics import model as model
import gc
import torch.cuda.nvtx as nvtx

import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from cs336_basics.nn_utils import softmax

# from execute_util import text, link, image
# from file_util import ensure_directory_exists
# from lecture_util import article_link
# from torch_util import get_device
# from lecture_06_utils import check_equal, check_equal2, get_local_url, round1, mean
import os


def gen_model_parameters(size:str):
    vocab_size = 1000
    d_model = 0
    d_ff = 0
    num_layers = 0
    num_heads = 0
    if size == 'small':
        d_model, d_ff, num_layers, num_heads = (768, 3072, 12, 12)
    elif size == 'medium':
        d_model, d_ff, num_layers, num_heads = (1024, 4096, 24, 16)
    elif size == 'large':
        d_model, d_ff, num_layers, num_heads = (1280, 5120, 36, 20)
    elif size == 'xl':
        d_model, d_ff, num_layers, num_heads = (1600, 6400, 48, 25)
    elif size == '2.7B':
        d_model, d_ff, num_layers, num_heads = (2560, 10240, 32, 32)
    else:
        raise ValueError(f"Unknown model size: {size}. Choose from ['small', 'medium', 'large', 'xl', '2.7B']")
    
    return vocab_size, d_model, d_ff, num_layers, num_heads

def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output

def run_transformers(model_size:str, context_length:int, num_steps:int, batch_size:int) -> Callable:
    torch.cuda.empty_cache()
    gc.collect()
    #define a model 
    with nvtx.range("define model"):
        vocab_size, d_model, d_ff, num_layers, num_heads = gen_model_parameters(size=model_size)
        basic_transformer = model.BasicsTransformerLM(vocab_size = vocab_size, 
                                                    context_length = context_length, 
                                                    d_model = d_model, 
                                                    num_layers = num_layers, 
                                                    num_heads = num_heads, 
                                                    d_ff = d_ff, 
                                                    rope_theta = 10000.
                                                    ).to(device='cuda')

    # Define an input(random)
    low = 0
    high = vocab_size
    x = torch.randint(low=low, high=high, size=(batch_size, context_length)).to(device='cuda')
    print(f'random input is {x}')


    def run():
        for step in range(num_steps):
            # Forward
            basic_transformer.zero_grad()
            with nvtx.range(f"step:{step}"):
                with nvtx.range("forward"):
                    y = basic_transformer(x).mean()

                # # Backward
                # with nvtx.range("backward"):
                #     y.backward()
            del y
            torch.cuda.empty_cache()
        
    return run 

def benchmark(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 10):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    
    # #warm up 
    with nvtx.range("warm up"):
        for _ in range(num_warmups):
            run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now
    with torch.autocast(device_type='cuda',dtype = torch.bfloat16):
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        times: list[float] = [] # @inspect times, @inspect description
        for trial in range(num_trials):  # Do it multiple times to capture variance
            with nvtx.range(f"trail:{trial}"):
                start_time = time.time()
                run()  # Actually perform computation

                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

                end_time = time.time()
                times.append((end_time - start_time) * 1000) # @inspect times
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        mean_time = sum(times)/len(times) # @inspect mean_time
    return times, mean_time

def benchmarking():
    '''
    finish this like https://stanford-cs336.github.io/spring2025-lectures/?trace=var%2Ftraces%2Flecture_06.json&step=422
    '''
    # init basic transformer with hyper parameters

    # chagable context length
    context_length = 128

    #fixed_params
    num_steps = 2
    batch_size = 1

    model_size_results = []
    # model_size_list = ['small', 'medium', 'large', 'xl', '2.7B']
    model_size_list = ['2.7B']
    for model_size in (model_size_list):
        print(f'model_size is {model_size}')
        #测试前清理内存
        torch.cuda.empty_cache()
        gc.collect()

        details, result = benchmark(f'run_transformers with {model_size}', 
                           run_transformers(model_size=model_size, context_length=context_length, num_steps=num_steps, batch_size=batch_size))
        model_size_results.append((model_size, result, details))

        torch.cuda.empty_cache()
        gc.collect()

    for result in model_size_results:
        print(result)

def benchmarking_memory():
    context_length_list = [128, 256, 512]
    num_steps = 1
    batch_size = 1
    model_size = '2.7B'
    context_length_results = []
    for context_length in context_length_list:
        print(f'context_length is:{context_length}')
        details, result = benchmark(f'run_transformers with {model_size}', 
                           run_transformers(model_size=model_size, context_length=context_length, num_steps=num_steps, batch_size=batch_size))  
        context_length_results.append((context_length, result, details))

if __name__ == "__main__":
    print(f'starting')
    model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    # benchmarking()
    benchmarking_memory()