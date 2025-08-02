import pandas as pd

df = pd.read_csv('../../data/sft/Qwen_Qwen3-4B/processing_summary.csv')
df2 = pd.read_csv('../../data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-07-31T22:55:24_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_epoch_5_step_final/processing_summary.csv')

print("Qwen3 4B:", df['success_failure'].value_counts(normalize=True))
print("Qwen3 4B Finetuned:", df2['success_failure'].value_counts(normalize=True))

print("Qwen3 4B Average turn count for successful attempts:", df[df['success_failure'] == "SUCCESS"].turn_count.mean())
print("Qwen3 4B Finetuned Average turn count for successful attempts:", df2[df2['success_failure'] == "SUCCESS"].turn_count.mean())

print("Qwen3 4B Average retries for successful attempts:", df[df['success_failure'] == "SUCCESS"].retry_count.mean())
print("Qwen3 4B Finetuned Average retries for successful attempts:", df2[df2['success_failure'] == "SUCCESS"].retry_count.mean())