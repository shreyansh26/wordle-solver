import os
import pandas as pd
from collections import Counter

files = os.listdir("../../data/sft/moonshotai_Kimi-K2-Instruct")
words = []
for file in files:
    if 'wordle_data' in file:
        word = file.split('wordle_data_')[1].lower()
        words.append(word)

ctr = Counter(words)

for k, v in ctr.items():
    if v > 1:
        print(k)
        df_lower = pd.read_csv(f'../../data/sft/moonshotai_Kimi-K2-Instruct/wordle_data_{k[:-4]}.csv')
        # print(df_lower.shape)
        # print(df_lower.iloc[-1]['curr_answer'])
        # print("-"*50)
        df_upper = pd.read_csv(f'../../data/sft/moonshotai_Kimi-K2-Instruct/wordle_data_{k.upper()[:-4]}.csv')
        # print(df_upper.shape)
        # print(df_upper.iloc[-1]['curr_answer'])
        # Remove longer
        if df_lower.shape >= df_upper.shape:
            print(f"rm wordle_data_{k[:-4]}.csv")
        else:
            print(f"rm wordle_data_{k.upper()[:-4]}.csv")
        # print("*"*100)
