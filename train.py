"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).
Features:
- Graceful shutdown on SIGINT / SIGTERM
- Rolling checkpoints every `save_interval` steps
- Shard-aware sequential data loader for training
- Distributed evaluation across all ranks
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
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

save_interval = 1
keep_best = True

wandb_log = False
wandb_project = 'gpt2-70m'
wandb_run_name = 'gpt2-70m'

dataset = 'data_mix'

gradient_accumulation_steps = 5
batch_size = 24
block_size = 1024

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

backend = 'nccl'

device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

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
    ddp_rank = 0

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
# Shard-aware data loading
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


def _init_train_loader():
    global _train_shards, _train_shard_lens, _train_cum_lens, _train_current_idx

    _train_shards = _list_train_shards()
    _train_shard_lens = [os.path.getsize(p) // 2 for p in _train_shards]
    total_tokens = sum(_train_shard_lens)

    tokens_per_rank = total_tokens // ddp_world_size
    rank_start = ddp_rank * tokens_per_rank

    _train_current_idx = rank_start
    _train_cum_lens = np.cumsum([0] + _train_shard_lens)

    if master_process:
        print(f"Sequential train loader: {total_tokens:,} tokens, "
              f"rank {ddp_rank} starts at index {rank_start:,}")


def _get_batch_random(split):
    """Random sampling used during evaluation (does not disturb sequential pointer)."""
    if split == 'val':
        data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    else:
        global _train_shards, _train_shard_lens
        if _train_shards is None:
            _train_shards = _list_train_shards()
            _train_shard_lens = [os.path.getsize(p) // 2 for p in _train_shards]

        lens = np.array(_train_shard_lens, dtype=np.int64)
        total = int(lens.sum())
        global_ix = np.random.randint(0, total - block_size - 1, size=batch_size)
        cum = np.cumsum(lens)
        starts = np.concatenate([[0], cum[:-1]])
        shard_ids = np.searchsorted(cum, global_ix, side='right')
        local_ix = global_ix - starts[shard_ids]

        buffers = [None] * batch_size
        order = np.argsort(shard_ids, kind='stable')
        i = 0
        while i < batch_size:
            sid = int(shard_ids[order[i]])
            j = i
            while j < batch_size and shard_ids[order[j]] == sid:
                j += 1
            data = np.memmap(_train_shards[sid], dtype=np.uint16, mode='r')
            max_off = _train_shard_lens[sid] - block_size - 1
            for idx in order[i:j]:
                off = min(int(local_ix[idx]), max_off)
                xb = torch.from_numpy((data[off:off + block_size]).astype(np.int64))
                yb = torch.from_numpy((data[off + 1:off + 1 + block_size]).astype(np.int64))
                buffers[int(idx)] = (xb, yb)
            del data
            i = j
        x = torch.stack([b[0] for b in buffers])
        y = torch.stack([b[1] for b in buffers])
    return x, y


def _get_batch_sequential():
    """Sequential loading for training (advances the per-rank pointer)."""
    global _train_shards, _train_shard_lens, _train_cum_lens, _train_current_idx
    global _train_shard_data, _train_current_shard_id

    if _train_shards is None:
        _init_train_loader()

    x_list = []
    y_list = []

    total_tokens = sum(_train_shard_lens)
    tokens_per_rank = total_tokens // ddp_world_size
    rank_start = ddp_rank * tokens_per_rank
    rank_end = (ddp_rank + 1) * tokens_per_rank

    for _ in range(batch_size):
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

            tokens[read_so_far:read_so_far + to_read] = \
                _train_shard_data[local_offset:local_offset + to_read].astype(np.int64)
            read_so_far += to_read
            curr_idx += to_read

            if curr_idx >= rank_end:
                curr_idx = rank_start

        x_list.append(tokens[:-1])
        y_list.append(tokens[1:])

        _train_current_idx += block_size
        if _train_current_idx >= rank_end:
            _train_current_idx = rank_start

    x = torch.stack([torch.from_numpy(s) for s in x_list])
    y = torch.stack([torch.from_numpy(s) for s in y_list])
    return x, y


def get_batch(split, sequential=True):
    """
    If sequential=True (training), reads contiguous chunks via the per-rank pointer.
    If sequential=False (evaluation), uses random sampling.
    """
    if not sequential or split == 'val':
        x, y = _get_batch_random(split)
    else:
        x, y = _get_batch_sequential()

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9
_last_ckpt_path = None


def find_latest_checkpoint():
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


shutdown_requested = False


def _signal_handler(signum, frame):
    global shutdown_requested
    if shutdown_requested:
        print("\nSecond signal received - forcing exit.")
        os._exit(1)
    shutdown_requested = True
    print(f"\n[signal {signum}] graceful shutdown requested...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# -----------------------------------------------------------------------------
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

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
    checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model


# -----------------------------------------------------------------------------
# Distributed evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """Evaluate on all ranks in parallel, then all_reduce to get global mean."""
    out = {}
    model.eval()
    local_iters = max(1, eval_iters // ddp_world_size)

    for split in ['train', 'val']:
        losses = torch.zeros(local_iters, device=device)
        for k in range(local_iters):
            X, Y = get_batch(split, sequential=False)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        local_mean = losses.mean()

        if ddp:
            all_reduce(local_mean, op=ReduceOp.SUM)
            local_mean /= ddp_world_size

        out[split] = local_mean.item()

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
# Training loop
# -----------------------------------------------------------------------------
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

try:
    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- Distributed evaluation (ALL ranks participate) ----
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            if master_process:
                print(f"step {iter_num}: train loss {losses['train']:.4f}, "
                      f"val loss {losses['val']:.4f}")
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
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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

        if iter_num % save_interval == 0:
            save_checkpoint(iter_num, best_val_loss)

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