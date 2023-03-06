from typing import Tuple

import os
import time
import json
from pathlib import Path

import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama.generation import LLaMA
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer





llama_weight_path = "./model_size/7B"
tokenizer_weight_path = "./model_size/"


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    print('Setup parallel complete!')

    torch.manual_seed(1)
    return local_rank, world_size


def get_pretrained_models(
        ckpt_path: str,
        tokenizer_path: str,
        local_rank: int,
        world_size: int) -> LLaMA:

    start_time = time.time()
    checkpoints = sorted(Path(llama_weight_path).glob("*.pth"))

    llama_ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(llama_ckpt_path, map_location="cpu")
    with open(Path(llama_weight_path) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1024, max_batch_size=64, **params)
    tokenizer = Tokenizer(model_path=f"{tokenizer_weight_path}/tokenizer.model")
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model.load_state_dict(checkpoint, strict=False)

    

    generator = LLaMA(model, tokenizer)
    print(f"Loaded done in {time.time() - start_time:.2f} seconds")
    return generator


def get_output(
        generator: LLaMA,
        prompt: str,
        max_gen_len: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95):
    prompts = [prompt]
    results = generator.generate(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p
    )

    return results