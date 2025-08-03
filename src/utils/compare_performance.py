import pandas as pd

def get_success_failure(df):
    return df['success_failure'].value_counts(normalize=True)

def get_average_turn_count(df):
    return df[df['success_failure'] == "SUCCESS"].turn_count.mean()

def get_average_retries(df):
    return df[df['success_failure'] == "SUCCESS"].retry_count.mean()

if __name__ == "__main__":
    df = pd.read_csv('../../data/sft/Qwen_Qwen3-4B/processing_summary.csv')
    df2 = pd.read_csv('../../data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-07-31T22:55:24_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_epoch_5_step_final/processing_summary.csv')
    df3 = pd.read_csv('../../data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-03T09:51:06_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_epoch_5_step_final/processing_summary.csv')
    df4 = pd.read_csv('../../data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-02T18:35:41_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_epoch_5_step_final/processing_summary.csv')
    df5 = pd.read_csv('../../data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-02T22:07:18_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_epoch_5_step_final/processing_summary.csv')

    print("Qwen3 4B:", get_success_failure(df))
    print("Qwen3 4B Finetuned LR=3e-5:", get_success_failure(df2))
    print("Qwen3 4B Finetuned LR=3e-5 + WSD:", get_success_failure(df3))
    print("Qwen3 4B Finetuned LR=5e-5:", get_success_failure(df4))
    print("Qwen3 4B Finetuned LR=5e-5 + WSD:", get_success_failure(df5))

    print("Qwen3 4B Average turn count for successful attempts:", get_average_turn_count(df))
    print("Qwen3 4B Finetuned LR=3e-5 Average turn count for successful attempts:", get_average_turn_count(df2))
    print("Qwen3 4B Finetuned LR=3e-5 + WSD Average turn count for successful attempts:", get_average_turn_count(df3))
    print("Qwen3 4B Finetuned LR=5e-5 Average turn count for successful attempts:", get_average_turn_count(df4))
    print("Qwen3 4B Finetuned LR=5e-5 + WSD Average turn count for successful attempts:", get_average_turn_count(df5))

    print("Qwen3 4B Average retries for successful attempts:", get_average_retries(df))
    print("Qwen3 4B Finetuned LR=3e-5 Average retries for successful attempts:", get_average_retries(df2))
    print("Qwen3 4B Finetuned LR=3e-5 + WSD Average retries for successful attempts:", get_average_retries(df3))
    print("Qwen3 4B Finetuned LR=5e-5 Average retries for successful attempts:", get_average_retries(df4))
    print("Qwen3 4B Finetuned LR=5e-5 + WSD Average retries for successful attempts:", get_average_retries(df5))