torchrun --nproc_per_node=2 load_llama3_fsdp_ckpt.py \
  --ckpt_dir /mnt/ssd2/shreyansh/models/llama32/exp_2025-08-21T16:07:26_llama32_3b_fsdp_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft/epoch_5/step_1101 \
  --params_json ./llama_3b_instruct/original/params.json \
  --max_seq_len 131072 \
  --export_hf_dir /mnt/ssd2/shreyansh/models/llama32/exp_2025-08-21T16:07:26_llama32_3b_fsdp_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft/epoch_5/merged_hf \
  --hf_model_name meta-llama/Llama-3.2-3B-Instruct