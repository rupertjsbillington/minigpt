# config for training GPT-2 (70M) to replicate codelion/gpt-2-70m
# launch: $ python train.py config/train_gpt2_70m.py
#     or: $ torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_70m.py

# wandb logging
wandb_log = True
wandb_project = 'gpt2-70m'
wandb_run_name = 'gpt2-70m-1B-tokens'

# data: tokenized 50/30/20 mix produced by /data/data_mix/prepare.py
dataset = 'data_mix'

# model architecture (matching codelion/gpt-2-70m config.json)
n_layer = 12
n_head = 8
n_embd = 512
bias = True
dropout = 0.1

# effective batch: 4 * 32 * 1024 = 131,072 tokens/iter
batch_size = 4
block_size = 1024
gradient_accumulation_steps = 32

# 1B tokens / 131,072 tokens per iter = 8,160 iterations
max_iters = 8160
lr_decay_iters = 8160

# learning rate schedule (cosine decay)
learning_rate = 5e-4
min_lr = 5e-5
warmup_iters = 163           # ~2% of total

# optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# checkpointing behaviour
save_interval = 500          # save every 500 steps; previous step file is deleted
keep_best = True             # keep a non-deleted best.pt on val-loss improvement

# eval stuff
eval_interval = 500
eval_iters = 512
log_interval = 1

# system
dtype = 'bfloat16'
compile = True