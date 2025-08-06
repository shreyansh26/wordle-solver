import os
import pandas as pd

# DATASET_DIR = "../data/sft/moonshotai_Kimi-K2-Instruct"
# DATASET_DIR = "../data/sft/deepseek-ai_DeepSeek-R1-0528"
DATASET_DIR = "../data/sft/openai_gpt-oss-120b"

files = os.listdir(DATASET_DIR)

rows = []

for file in files:
    if 'wordle_data' not in file:
        continue
    df_t = pd.read_csv(os.path.join(DATASET_DIR, file), dtype=str)
    df_t['turn'] = df_t['turn'].astype(int)
    row = {
        "word": str(df_t['correct_answer'].iloc[0]),
        "num_rows": df_t.shape[0],
        "is_successful": df_t['curr_answer'].iloc[-1]
    }
    rows.append(row)

df = pd.DataFrame(rows)
# df.to_csv('../data/sft/train/moonshot_kimi_k2_summary_v2.csv', index=False)
# df.to_csv('../data/sft/train/deepseek_r1_summary.csv', index=False)
df.to_csv('../data/sft/train/openai_gpt_oss-120b_summary.csv', index=False)