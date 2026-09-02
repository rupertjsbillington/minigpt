"""
TorchTPU Training Script for GPT-2 70M on TPU v5e-8.
Optimized for PrivateUse1 interface, Fused Eager mode, and 128-dim attention heads.
"""
import os
import time
import math
import glob
import signal
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

save_interval = 500  # Checkpointing every step is too slow; 500 is standard
keep_best = True

wandb_log = False
wandb_project = 'gpt2-70m-tpu'
wandb_run_name = 'run'

dataset = 'data_mix'

# Optimized for 8 chips: 8 * 1 * 15 = 120 sequences global batch (matches original)
gradient_accumulation_steps = 8 
batch_size = 15
block_size = 1024

# TPU Optimized Model Architecture:
# n_embd=512, n_head=4 => head_dim=128 (Peak TPU TensorCore efficiency)
n_layer = 12
n_head = 4
n_embd = 512
dropout = 0.0
bias = True

learning_rate = 5e-4
max_iters = 8138
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 162
lr_decay_iters = 8138
min_lr = 5e-5

backend = 'gloo' # Gloo for control plane, TorchTPU handles tensor comms

# System
device = 'tpu'
dtype = 'bfloat16' # TPUs prefer BF16; no GradScaler needed
compile = True     # Maps to XLA/StableHLO via TorchTPU

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Distributed Setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    dist.init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    # PrivateUse1 device mapping
    device = f'tpu:{ddp_local_rank}'
    
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

# TPU Autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type='tpu', dtype=ptdtype)

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Standard device transfer works asynchronously with TorchTPU
    x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Model Init
# -----------------------------------------------------------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    # Resume logic omitted for brevity, standard torch.load applies
    pass
else:
    print(f"Initializing from {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))

model.to(device)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

if compile:
    print("compiling the model... (TorchTPU maps this to XLA/StableHLO)")
    model = torch.compile(model)

if ddp:
    model = DDP(model)

raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()
iter_num = 0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            # Save checkpoint logic...

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # Fused Eager mode handles the graph boundaries automatically
        loss.backward()
        X, Y = get_batch('train')

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    dist.destroy_process_group()