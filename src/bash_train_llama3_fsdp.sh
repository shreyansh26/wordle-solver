set -x
torchrun --nnodes=1 --nproc-per-node=2 --master_port 29500 run_train_llama3_fsdp.py --async-dcp