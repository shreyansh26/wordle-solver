import os
import pandas as pd

DATASET_DIR = "../data/sft/moonshotai_Kimi-K2-Instruct"

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
df.to_csv('../data/sft/train/moonshot_kimi_k2_summary.csv', index=False)