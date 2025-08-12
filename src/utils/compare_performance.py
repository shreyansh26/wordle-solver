import pandas as pd

def get_success_failure(df):
    return df['success_failure'].value_counts(normalize=True)

def get_average_turn_count(df):
    return df[df['success_failure'] == "SUCCESS"].turn_count.mean()

def get_average_retries(df):
    return df[df['success_failure'] == "SUCCESS"].retry_count.mean()

if __name__ == "__main__":
    df = pd.read_csv('../../data/sft/Qwen_Qwen3-4B/processing_summary.csv')
    df2 = pd.read_csv('../../data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-02T18:35:41_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2_epoch_5_step_final/processing_summary.csv')
    df3 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-07T23:32:38_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft_epoch_5_step_final/processing_summary.csv')
    df4 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-08T00:44:45_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft_epoch_5_step_final/processing_summary.csv')
    df5 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_exp_2025-08-08T00:53:20_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2_v2_sft_epoch_5_step_final/processing_summary.csv')
    df6 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_grpo_vllm_rl_v2_checkpoint-40/processing_summary.csv')
    df7 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_grpo_vllm_rl_v3_checkpoint-150/processing_summary.csv')
    df8 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_grpo_vllm_rl_v4_checkpoint-686/processing_summary.csv')
    df9 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_grpo_vllm_rl_v5_checkpoint-300/processing_summary.csv')
    df10 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_grpo_vllm_rl_v6_checkpoint-686/processing_summary.csv')
    df11 = pd.read_csv('/mnt/ssd1/shreyansh/home_dir/wordle_grpo/data/sft/_mnt_ssd2_shreyansh_models_qwen3_grpo_vllm_rl_v7_checkpoint-686/processing_summary.csv')

    print("Qwen3 4B:", get_success_failure(df))
    print("Qwen3 4B Finetuned Kimi K2 LR=5e-5:", get_success_failure(df2))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=5e-5:", get_success_failure(df3))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=3e-5:", get_success_failure(df4))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=7e-5:", get_success_failure(df5))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 4 Gens:", get_success_failure(df6))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 4 Gens:", get_success_failure(df7))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens:", get_success_failure(df8))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens cosine:", get_success_failure(df9))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens cosine + better rewards:", get_success_failure(df10))    
    print("Qwen3 4B Finetuned GRPO LR=3e-6; 8 Gens constant + better rewards:", get_success_failure(df11))

    print("Qwen3 4B Average turn count for successful attempts:", get_average_turn_count(df))
    print("Qwen3 4B Finetuned Kimi K2 LR=5e-5 Average turn count for successful attempts:", get_average_turn_count(df2))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=5e-5 Average turn count for successful attempts:", get_average_turn_count(df3))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=3e-5 Average turn count for successful attempts:", get_average_turn_count(df4))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=7e-5 Average turn count for successful attempts:", get_average_turn_count(df5))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 4 Gens Average turn count for successful attempts:", get_average_turn_count(df6))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 4 Gens Average turn count for successful attempts:", get_average_turn_count(df7))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens Average turn count for successful attempts:", get_average_turn_count(df8))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens cosine Average turn count for successful attempts:", get_average_turn_count(df9))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens cosine + better rewards Average turn count for successful attempts:", get_average_turn_count(df10))
    print("Qwen3 4B Finetuned GRPO LR=3e-6; 8 Gens constant + better rewards Average turn count for successful attempts:", get_average_turn_count(df11))

    print("Qwen3 4B Average retries for successful attempts:", get_average_retries(df))
    print("Qwen3 4B Finetuned Kimi K2 LR=5e-5 Average retries for successful attempts:", get_average_retries(df2))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=5e-5 Average retries for successful attempts:", get_average_retries(df3))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=3e-5 Average retries for successful attempts:", get_average_retries(df4))
    print("Qwen3 4B Finetuned Kimi K2 v2 LR=7e-5 Average retries for successful attempts:", get_average_retries(df5))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 4 Gens Average retries for successful attempts:", get_average_retries(df6))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 4 Gens Average retries for successful attempts:", get_average_retries(df7))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens Average retries for successful attempts:", get_average_retries(df8))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens cosine Average retries for successful attempts:", get_average_retries(df9))
    print("Qwen3 4B Finetuned GRPO LR=1e-6; 8 Gens cosine + better rewards Average retries for successful attempts:", get_average_retries(df10))
    print("Qwen3 4B Finetuned GRPO LR=3e-6; 8 Gens constant + better rewards Average retries for successful attempts:", get_average_retries(df11))