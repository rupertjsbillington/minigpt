"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
Features added for the 70M replication run:
Graceful shutdown on SIGINT / SIGTERM (finishes the current step, saves, exits).
Saves a checkpoint every `save_interval` steps (default: every step).
Rolling checkpoints: the previous step's file is deleted when a new one is saved.
Shard-aware data loader reading data/<dataset>/train_shards/*.bin
Optimized for sequential I/O to prevent storage bottlenecks.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2-70m on the mixed 1B dataset
# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False            # if True, script exits right after the first eval
always_save_checkpoint = True
init_from = 'scratch'        # 'scratch' | 'resume' | 'gpt2*' | HF model id

# checkpointing behaviour
save_interval = 1            # save a checkpoint every N steps (1 = every step)
keep_best = True             # additionally keep a non-deleted best.pt

# wandb logging
wandb_log = False
wandb_project = 'gpt2-70m'
wandb_run_name = 'gpt2-70m'

# data
dataset = 'data_mix'

# batch / sequence
# Adjusted for ~24GB VRAM GPUs (4 * 30 = 120 effective micro-batches per step)
gradient_accumulation_steps = 30
batch_size = 4               # micro-batch size
block_size = 1024

# model (70M defaults matching codelion/gpt-2-70m)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1
bias = True

# adamw optimizer
learning_rate = 5e-4
max_iters = 8113
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 162
lr_decay_iters = 8113
min_lr = 5e-5

# DDP settings
backend = 'nccl'

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
# -----------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0  # Needed for sequential loader partition logic on single GPU

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Shard-aware sequential data loading
# -----------------------------------------------------------------------------
data_dir = os.path.join('data', dataset)
val_bin_path = os.path.join(data_dir, 'val.bin')

_train_shards = None
_train_shard_lens = None
_train_cum_lens = None
_train_current_idx = None
_train_shard_data = None
_train_current_shard_id = -1

def _list_train_shards():
    """Return a list of train shard .bin paths (or a single train.bin fallback)."""
    shard_dir = os.path.join(data_dir, 'train_shards')
    if os.path.isdir(shard_dir):
        shards = sorted(glob.glob(os.path.join(shard_dir, '*.bin')))
        if shards:
            return shards
    single = os.path.join(data_dir, 'train.bin')
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(
        f"No train shards found. Looked in '{shard_dir}' and for '{single}'. "
        f"Run prepare.py first.")

def init_train_loader():
    global _train_shards, _train_shard_lens, _train_cum_lens, _train_current_idx
    
    _train_shards = _list_train_shards()
    _train_shard_lens = [os.path.getsize(p) // 2 for p in _train_shards]
    total_tokens = sum(_train_shard_lens)
    
    # Partition dataset among DDP ranks to avoid duplicate work
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
        # val is a single contiguous .bin file; random sampling is fine for eval
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
                
            seq_len = block_size + 1
            tokens = np.empty(seq_len, dtype=np.int64)
            read_so_far = 0
            curr_idx = _train_current_idx
            
            while read_so_far < seq_len:
                shard_id = np.searchsorted(_train_cum_lens[1:], curr_idx, side='right')
                local_offset = curr_idx - _train_cum_lens[shard_id]
                
                if shard_id != _train_current_shard_id:
                    _train_current_shard_id = shard_id
                    _train_shard_data = np.memmap(_train_shards[shard_id], dtype=np.uint16, mode='r')
                
                available = _train_shard_lens[shard_id] - local_offset
                to_read = min(seq_len - read_so_far, available)
                
                tokens[read_so_far : read_so_far + to_read] = _train_shard_data[local_offset : local_offset + to_read].astype(np.int64)
                read_so_far += to_read
                curr_idx += to_read
                
                if curr_idx >= rank_end:
                    curr_idx = rank_start
            
            x_list.append(tokens[:-1])
            y_list.append(tokens[1:])
            
            _train_current_idx += block_size
            if _train_current_idx >= rank_end:
                _train_current_idx = rank_start
                
        x = torch.stack([torch.from_numpy(seq) for seq in x_list])
        y = torch.stack([torch.from_numpy(seq) for seq in y_list])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# -----------------------------------------------------------------------------
# Checkpoint helpers (rolling + graceful save)
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9
_last_ckpt_path = None  # tracks the rolling latest checkpoint for deletion

def find_latest_checkpoint():
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
    """Save a rolling checkpoint for `step`, deleting the previous one."""
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

    ckpt_path = os.path.join(out_dir, f'ckpt_{step:08d}.pt')
    tmp_path = ckpt_path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, ckpt_path)

    if _last_ckpt_path is not None and _last_ckpt_path != ckpt_path \
            and os.path.exists(_last_ckpt_path):
        os.remove(_last_ckpt_path)
    _last_ckpt_path = ckpt_path

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
# -----------------------------------------------------------------------------
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
    ckpt_path = find_latest_checkpoint()
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

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # free memory

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# Training Loop helpers
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

# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

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

        # forward / backward / update with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # timing / logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            if wandb_log:
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
            if iter_num % save_interval != 0:
                save_checkpoint(iter_num, best_val_loss)
            print(f"Graceful shutdown complete. Checkpoint saved at step {iter_num}.")
            break

        if iter_num > max_iters:
            break

except KeyboardInterrupt:
    print("Interrupted - saving checkpoint before exit...")
    save_checkpoint(iter_num, best_val_loss)

except Exception as e:
    print(f"Unexpected error: {e}\nSaving checkpoint before exit...")
    save_checkpoint(iter_num, best_val_loss)
    raise

finally:
    if ddp:
        try:
            destroy_process_group()
        except Exception:
            pass