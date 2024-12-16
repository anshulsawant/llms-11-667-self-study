import itertools
import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable
from rich import print

import copy
import argparse
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lm.model import DecoderLM
from lm.utils import (
    count_params,
    determine_device,
    enable_tf32,
    estimate_model_disk_size,
)


def random_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """An infinite generator that samples batches of sequences from the tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        An infinite generator that samples batches of sequences from the
        tokens. Each batch has shape (B x S). Every sequence in the batch is
        a contiguous subsequence of x, sampled uniformly at random. The
        output tensor should be on the right device.
    """

    max_idx = len(tokens) - seq_len 
    while True:
        start_indices = torch.randint(0, max_idx + 1, (batch_size,))
        batch = torch.stack([tokens[i:i + seq_len] for i in start_indices]).to(device)
        yield batch


def sequential_batch_sampler(
    tokens: torch.LongTensor, device: str, batch_size: int, seq_len: int
) -> Iterator[torch.LongTensor]:
    """A generator that yields batches of tokens.

    Args:
        tokens: a 1d torch tensor of token ids
        device: the device to put the batch on
        batch_size: the batch size of the output tensor (B)
        seq_len: the sequence length of the output tensor (S)

    Returns:
        A generator that yields a batch of tokens at a time. Each batch has
        shape (B x S). Every sequence in the batch is a contiguous subsequence
        of x in sequential order. The output tensor should be on the right
        device.

    Note: If the last batch is incomplete, which could happen when the number
        of tokens is not divisible by (batch_size * seq_len), you could drop
        the last batch.
    """

    n_tokens = batch_size * seq_len
    n_batches = len(tokens) // n_tokens
    for batch in range(n_batches):
        yield tokens[batch * n_tokens : (batch + 1) * n_tokens
                     ].reshape((batch_size, seq_len)).to(device=device)
            
def cosine_lr_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr: float,
    max_lr: float,
) -> Callable[[int], float]:
    def get_lr(t: int) -> float:
        """Outputs the learning rate at step t under the cosine schedule.

        Args:
            t: the current step number

        Returns:
            lr: learning rate at step t

        Hint: Question 3.2
        """

        assert max_lr >= min_lr >= 0.0
        assert num_training_steps >= num_warmup_steps >= 0

        if t <= num_warmup_steps:
            lr = max_lr * t/(num_warmup_steps)
        elif t >= num_training_steps:
            lr = min_lr
        else:
            theta = math.pi * (t - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            lr = min_lr + (math.cos(theta) + 1) * (max_lr - min_lr)/2
        return lr

    return get_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def compute_language_modeling_loss(
    input_ids: torch.LongTensor, logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Outputs the language modeling loss given input_ids and logits

    Args:
        input_ids: the input token ids B x S
        logits: the next token logits produced by the language model B x S x V

    Returns:
        loss: the mean cross entropy loss for next token prediction

    Hint: Think about what are the groundtruth labels for next token prediction.
    """
    batch_size = input_ids.size()[0]
    seq_len = input_ids.size()[1]
    labels = input_ids[:, 1:]
    shifted_logits = logits[:, :-1 , :]
    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    device = input_ids.device
    batch_index = torch.tensor(range(batch_size)).repeat_interleave(seq_len - 1).to(device=device)
    seq_index = torch.tensor(range(seq_len-1)).repeat(batch_size).to(device=device)
    return -torch.mean(log_probs[batch_index, seq_index, labels.flatten()])


def train(
        model: DecoderLM,
        batch_sampler: Iterator[torch.LongTensor],
        optimizer: torch.optim.Optimizer,
        lr_schedule: Callable[[int], float],
        config, 
        autocast: torch.autocast | nullcontext = nullcontext(),
) -> None:
    """A training loop for the language model

    Args:
        model: the decoder LM
        batch_sampler: a generator that produces batches of token ids
        optimizer: an optimizer for gradient update
        lr_schedule: a callable that produces the learning at a step number
        autocast: a context manager that handles tensor casting (you do not need
          to care about this for your implementation)
        num_training_steps: number of steps to train for
        grad_accumulation_steps: number of "micro" training steps before each
          gradient update
    """
    # stores training losses for the 20 latest steps
    grad_accumulation_steps = config.grad_accumulation_steps
    num_training_steps = config.num_training_steps
    early_stopping = config.get('early_stopping', False)
    
    losses = deque(maxlen=20 * grad_accumulation_steps)

    for step in (pbar := trange(num_training_steps)):
        t0 = time.time()
        lr = lr_schedule(step)
        set_lr(optimizer, lr)

        for _ in range(grad_accumulation_steps):
            input_ids = next(batch_sampler)
            with autocast:
                logits = model(input_ids)
            loss = compute_language_modeling_loss(input_ids, logits)
            (loss / grad_accumulation_steps).backward()
            loss_f = loss.item()
            losses.append(loss_f)

        optimizer.step()
        loss_mean = np.mean(losses).item()

        FLOPs_per_step = (
            model.flops_per_token
            * input_ids.shape[0]
            * input_ids.shape[1]
            * grad_accumulation_steps
        )
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        pbar.set_postfix(
            {
                "train loss": f"{loss_mean:.2f}",
                "TFLOPS": f"{FLOPs_per_step / dt / 1e12:.1f}",
            }
        )
        wandb.log({"train-loss": loss_mean, "learning-rate": lr}, step=step)

        stop_now = (config.early_stopping_loss <= loss_mean and
                    config.early_stopping_min_steps <= step) if early_stopping else False
        if stop_now:
            print(f"Training loss too high. Loss is {loss_mean}. Stopping early after {step} steps.")
            return


@torch.inference_mode()
def evaluate(
        model: DecoderLM,
        batch_sampler: Iterator[torch.LongTensor],
        autocast: torch.autocast | nullcontext = nullcontext(),
        test_run = False,
) -> dict[str, float]:
    losses = []

    i = 0
    for input_ids in tqdm(batch_sampler, desc="evaluating.."):
        with autocast:
            logits = model(input_ids)
        loss = compute_language_modeling_loss(input_ids, logits)
        losses.append(loss.item())
        i += 1
        if test_run and i >= 100:
            break;

    # mean of the losses is the average negative log likelihood
    mean_loss = sum(losses) / len(losses)
    perplexity = None
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError as e:
        perplexity = float('inf')

    eval_results = {
        "val-loss": mean_loss,
        "val-perplexity": perplexity,
    }
    wandb.log(eval_results)
    return eval_results


def training_run(config, train_tokens, val_tokens):
    max_flops = config.get('max_flops', None)
    num_training_steps = config.get('num_training_steps', None)
    run_no = config.get('run_no', None)
    tag = config.get('tag', None)
    name_prefix = config.get('name_prefix', None)
    name = None if name_prefix is None else name_prefix + str(run_no)
    early_stopping = config.get('early_stopping', False)
    early_stopping_loss = config.get('early_stopping_loss', None)
    early_stopping_min_steps = config.get('early_stopping_min_steps', None)
    save_model = config.get('save_model', True)
    test_run = config.get('test_run', False)
    swish = config.model_config.get('swish', False)
    config.model_config.swish = swish


    os.makedirs(config.output_dir, exist_ok=True)
    assert config.seq_len <= config.model_config.n_positions

    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    print(f"model parameters = {count_params(model) / 1e6:.0f}M")

    model_disk_size_MB = estimate_model_disk_size(model) * 1e-6
    if model_disk_size_MB > 98:
        print(
            f"[red]WARNING: your model is {model_disk_size_MB:.1f}MB. "
            "The largest model size allowed by GradeScope is 100MB, "
            "and you may have trouble with submitting the assignment. "
            "Please update your config so your model is at most 100 MB.[/red]"
        )
    else:
        print(
            f"Your model is {model_disk_size_MB:.1f}MB. This should be within "
            "the 100MB limit of Gradescope."
        )

    train_sampler = random_batch_sampler(
        train_tokens, device, config.batch_size, config.seq_len
    )
    val_sampler = sequential_batch_sampler(
        val_tokens, device, config.batch_size, config.seq_len
    )
    print(f"train dataset tokens = {len(train_tokens) / 1e6:.0f}M")
    
    if max_flops:
        num_training_steps = int(max_flops//(model.flops_per_token *
                                                config.grad_accumulation_steps *
                                                config.batch_size *
                                                config.seq_len))
        config.num_training_steps = num_training_steps
        print(f"max_flops specified."
              f" Overwriting num_training_steps with value {num_training_steps}.")

    FLOPs = (
        model.flops_per_token
        * num_training_steps
        * config.grad_accumulation_steps
        * config.batch_size
        * config.seq_len
    )
    print(f"train FLOPs = {FLOPs:.2e}")
    if FLOPs > 1e17:
        print(
            f"[red]WARNING: your train FLOPs is {FLOPs:.2e}. "
            "This is more than the max compute that we allow (1e+17). "
            "Please reduce your model size or train steps.[/red]"
        )

    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    print("#" * 40, OmegaConf.to_yaml(config).strip(), "#" * 40, sep="\n")

    # prepare optimizer and lr schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # will set this dynamically in the training loop
        betas=(0.9, 0.95),
        fused=device == "cuda",
    )
    lr_schedule = cosine_lr_schedule(
        config.num_warmup_steps, config.num_training_steps, config.min_lr, config.max_lr
    )
    autocast = (
        torch.autocast(
            device,
            dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
        )
        if device == "cuda"
        else nullcontext()
    )

    wandb.init(
        project="llms-hw2",
        name=name,
        config=OmegaConf.to_container(config),
        tags=[] if tag is None else [tag],
        reinit=True)
    
    # training
    model.train()
    train(
        model,
        train_sampler,
        optimizer,
        lr_schedule,
        config,
        autocast,
    )

    if save_model:
        # save the trained model
        model_path = os.path.join(config.output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"model saved to {model_path}")

    # evaluation
    model.eval()
    test_run = config.test_run if "test_run" in config else None
    eval_results = evaluate(model, val_sampler, autocast, test_run=test_run)
    wandb.finish()
    print("evaluation results:", json.dumps(eval_results))
    with open(os.path.join(config.output_dir, "eval.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    print("done!")
    return model, eval_results


def main(config_file, test_run):
    enable_tf32()

    config = OmegaConf.load(config_file)
    config.test_run = test_run

    tokens = np.load("data/tokens.npz")

    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))

    model, eval_results = training_run(config, train_tokens, val_tokens)
    print("done!")

def product_config(config):
    res = []
    keys = config.keys()
    for instance in itertools.product(*config.values()):
        res.append(OmegaConf.create(dict(zip(keys, instance))))
    return res


def build_sweep_config(config, hyper_param, output_dir, run_no):
    model_config = OmegaConf.create()
    model_config.n_embd = hyper_param.n_embd
    model_config.n_head = hyper_param.n_head
    model_config.n_positions = hyper_param.n_positions
    model_config.n_layer = hyper_param.n_layer
    model_config.swish = False

    
    c = copy.deepcopy(config) 
    c.model_config = model_config
    c.seq_len = model_config.n_positions
    c.run_no = run_no
    c.output_dir = os.path.join(output_dir, str(run_no))
    c.grad_accumulation_steps = 1
    c.min_lr = hyper_param.lr[0]
    c.max_lr = hyper_param.lr[1]
    
    return OmegaConf.merge(c, hyper_param)
    

def generate_configs(config_file):
    count = 0
    config = OmegaConf.load(config_file)
    hyper_params = product_config(config.parameters)
    

    indices = np.arange(len(hyper_params))

    if config.method == all:
        config.num_run = len(hyper_params)
        
    if config.method == "random":
        n_samples = min(config.num_run, len(hyper_params))
        indices = np.random.choice(indices, size=n_samples, replace=False)
    
    for i in indices:
        yield build_sweep_config(config, hyper_params[i], config.output_dir, count)
        count += 1
    
    
def sweep(config_file, test_run):
    enable_tf32()

    tokens = np.load("data/tokens.npz")
    train_tokens = torch.from_numpy(tokens["train"].astype(int))
    val_tokens = torch.from_numpy(tokens["val"].astype(int))

    for config in generate_configs(config_file):
        config.test_run = test_run
        model, eval_results = training_run(
            config,
            train_tokens,
            val_tokens)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help="Config file for sweep or a single run.")
    parser.add_argument("--sweep", help="Whether to run a hyperparam sweep", action="store_true")
    parser.add_argument("--test_run", help="Whether this is a test run", action="store_true")
    args = parser.parse_args()
    if args.sweep:
        sweep(args.config_file, args.test_run)
    else:
        main(args.config_file, args.test_run)
    
