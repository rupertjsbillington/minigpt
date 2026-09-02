"""
TorchTPU Training Script for GPT-2 70M on TPU v5e-8.
Optimized for PrivateUse1 interface, Fused Eager mode, and 64-dim attention heads.
"""
import os
import re
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
wandb_project = 'gpt2-70m'
wandb_run_name = 'gpt2-70m-1B-tokens'

dataset = 'data_mix'

# Optimized for 8 chips: 8 * 1 * 15 = 120 sequences global batch (matches original)
gradient_accumulation_steps = 8 
batch_size = 15
block_size = 1024

# TPU Optimized Model Architecture:
# n_embd=512, n_head=8 => head_dim=64 (Peak TPU TensorCore efficiency)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
bias = True

learning_rate = 5e-4
max_iters = 8113
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 162
lr_decay_iters = 8113
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
# Data (Optimized for Sequential I/O & DDP Partitioning)
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)
val_bin_path = os.path.join(data_dir, 'val.bin')

_train_shards = None
_train_shard_lens = None
_train_cum_lens = None
_train_current_idx = None
_train_shard_data = None
_train_current_shard_id = -1

def init_train_loader():
    global _train_shards, _train_shard_lens, _train_cum_lens, _train_current_idx
    
    shard_dir = os.path.join(data_dir, 'train_shards')
    if os.path.isdir(shard_dir):
        _train_shards = sorted(glob.glob(os.path.join(shard_dir, '*.bin')))
    else:
        _train_shards = [os.path.join(data_dir, 'train.bin')]
        
    if not _train_shards:
        raise FileNotFoundError(f"No train shards found in {shard_dir} or {data_dir}")
        
    _train_shard_lens = [os.path.getsize(p) // 2 for p in _train_shards]
    total_tokens = sum(_train_shard_lens)
    
    # Partition dataset among DDP ranks to avoid duplicate work and maximize throughput
    tokens_per_rank = total_tokens // ddp_world_size
    rank_start = ddp_rank * tokens_per_rank
    
    _train_current_idx = rank_start
    _train_cum_lens = np.cumsum([0] + _train_shard_lens)
    
    if master_process:
        print(f"Initialized sequential train loader. Total tokens: {total_tokens:,}")
        print(f"Rank {ddp_rank}: processing {tokens_per_rank:,} tokens starting from index {rank_start:,}")

def get_batch(split):
    global _train_shards, _train_shard_lens, _train_cum_lens, _train_current_idx, _train_shard_data, _train_current_shard_id

    if split == 'val':
        # Keep original random sampling for validation to ensure representative loss across the whole set
        data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    else:
        if _train_shards is None:
            init_train_loader()
            
        x_list = []
        y_list = []
        
        total_tokens = sum(_train_shard_lens)
        tokens_per_rank = total_tokens // ddp_world_size
        rank_start = ddp_rank * tokens_per_rank
        rank_end = (ddp_rank + 1) * tokens_per_rank
        
        for _ in range(batch_size):
            # Wrap around if the next sequence exceeds the rank's partition
            if _train_current_idx + block_size + 1 > rank_end:
                _train_current_idx = rank_start
                
            # Read sequence, handling shard boundaries seamlessly
            seq_len = block_size + 1
            tokens = np.empty(seq_len, dtype=np.int64)
            read_so_far = 0
            curr_idx = _train_current_idx
            
            while read_so_far < seq_len:
                # Find which shard the current index falls into
                shard_id = np.searchsorted(_train_cum_lens[1:], curr_idx, side='right')
                local_offset = curr_idx - _train_cum_lens[shard_id]
                
                # Open new memmap if we crossed into a new shard
                if shard_id != _train_current_shard_id:
                    _train_current_shard_id = shard_id
                    # Memmaps are memory-mapped; reassigning closes the old one via GC
                    _train_shard_data = np.memmap(_train_shards[shard_id], dtype=np.uint16, mode='r')
                
                available = _train_shard_lens[shard_id] - local_offset
                to_read = min(seq_len - read_so_far, available)
                
                tokens[read_so_far : read_so_far + to_read] = _train_shard_data[local_offset : local_offset + to_read].astype(np.int64)
                read_so_far += to_read
                curr_idx += to_read
                
                # Wrap around if we hit the end of the rank's partition mid-sequence
                if curr_idx >= rank_end:
                    curr_idx = rank_start
            
            x_list.append(tokens[:-1])
            y_list.append(tokens[1:])
            
            # Advance by block_size for non-overlapping contiguous chunks
            _train_current_idx += block_size
            if _train_current_idx >= rank_end:
                _train_current_idx = rank_start
                
        x = torch.stack([torch.from_numpy(seq) for seq in x_list])
        y = torch.stack([torch.from_numpy(seq) for seq in y_list])

    x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Checkpoint helpers (rolling + graceful save)
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9
_last_ckpt_path = None  # tracks the rolling latest checkpoint for deletion


def _find_latest_checkpoint():
    """Locate the most recent rolling checkpoint in out_dir."""
    candidates = [c for c in glob.glob(os.path.join(out_dir, 'ckpt_*.pt'))
                  if not c.endswith('.tmp')]
    if candidates:
        def step_of(p):
            m = re.search(r'ckpt_(\d+)\.pt$', os.path.basename(p))
            return int(m.group(1)) if m else -1
        return max(candidates, key=step_of)
    for legacy in ('ckpt.pt', 'best.pt'):
        p = os.path.join(out_dir, legacy)
        if os.path.exists(p):
            return p
    return None


def save_checkpoint(step, best_loss, is_best=False):
    """Save a rolling checkpoint for `step`, deleting the previous one.

    Writes to a temp file first and atomically renames so an interrupted save
    never leaves a corrupted checkpoint behind.
    """
    global _last_ckpt_path
    if not master_process:
        return

    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': step,
        'best_val_loss': best_loss,
        'config': config,
    }

    # rolling latest checkpoint
    ckpt_path = os.path.join(out_dir, f'ckpt_{step:08d}.pt')
    tmp_path = ckpt_path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, ckpt_path)  # atomic on POSIX

    # delete the previous rolling checkpoint
    if _last_ckpt_path is not None and _last_ckpt_path != ckpt_path \
            and os.path.exists(_last_ckpt_path):
        os.remove(_last_ckpt_path)
    _last_ckpt_path = ckpt_path

    # optionally keep a separate, never-deleted best checkpoint
    if keep_best and is_best:
        best_path = os.path.join(out_dir, 'best.pt')
        best_tmp = best_path + '.tmp'
        torch.save(checkpoint, best_tmp)
        os.replace(best_tmp, best_path)


# graceful shutdown handling
shutdown_requested = False


def _signal_handler(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        print("\nSecond signal received - forcing exit.")
        os._exit(1)
    shutdown_requested = True
    print(f"\n[signal {signum}] graceful shutdown requested: "
          f"finishing current step, saving checkpoint, then exiting...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
# -----------------------------------------------------------------------------

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50257")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50257
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    ckpt_path = _find_latest_checkpoint()
    assert ckpt_path is not None, f"No checkpoint found to resume from in {out_dir}"
    print(f"Resuming training from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    # init from a HuggingFace model (e.g. 'gpt2' or 'codelion/gpt-2-70m')
    print(f"Initializing from pretrained weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'tpu')
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free memory

if compile:
    print("compiling the model... (TorchTPU maps this to XLA/StableHLO)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

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
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model

try:
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # periodic evaluation
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(iter_num, best_val_loss, is_best=True)

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

        # timing / logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
            })

        iter_num += 1
        local_iter_num += 1

        # save every `save_interval` steps (rolling: previous checkpoint is deleted)
        if iter_num % save_interval == 0:
            save_checkpoint(iter_num, best_val_loss)

        # graceful shutdown check (after the step + save completed)
        if shutdown_requested:
            # Ensure the latest checkpoint is saved here if the last step was not a save step
            if iter_num % save_interval != 0:
                save_checkpoint(iter_num, best_val_loss)
            print(f"Graceful shutdown complete. Checkpoint saved at step {iter_num}.")
            break

        if iter_num > max_iters:
            break

except KeyboardInterrupt:
    # fallback in case a signal still surfaces as KeyboardInterrupt
    print("Interrupted - saving checkpoint before exit...")
    save_checkpoint(iter_num, best_val_loss)

except Exception as e:
    # save progress on unexpected errors too, then re-raise
    print(f"Unexpected error: {e}\nSaving checkpoint before exit...")
    save_checkpoint(iter_num, best_val_loss)
    raise

finally:
    if ddp:
        try:
            dist.destroy_process_group()
        except Exception:
            pass