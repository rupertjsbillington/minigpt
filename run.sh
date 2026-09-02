# PJRT_DEVICE=TPU is often still required by the underlying runtime to initialize chips
PJRT_DEVICE=TPU torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  train_tpu.py