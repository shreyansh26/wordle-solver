import os
import json
import ast
import random
import pandas as pd
from transformers import AutoTokenizer

random.seed(1337)

def get_formatted_training_example(row, tokenizer):
    messages = row['messages']
    assert isinstance(messages, list)

    response = row['response']
    assert isinstance(response, str)

    word = row['correct_answer']
    assert isinstance(word, str)

    turn = row['turn']
    assert isinstance(turn, int)

    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted_output = f"{response}{tokenizer.eos_token}"

    return {
        "instruction": formatted_input,
        "output": formatted_output,
        "word": word,
        "turn": turn,
    }

if __name__ == "__main__":
    DATASET_DIR = '../data/sft/moonshotai_Kimi-K2-Instruct'
    df = pd.read_csv('../data/sft/train/moonshot_kimi_k2_summary.csv', dtype={'word': str, 'num_rows': int, 'is_successful': str})
    df = df[df['is_successful'] == "SUCCESS"]
    print("Successful words:", df.shape[0])

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    records = []

    for word in df['word'].unique():
        df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
        df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
        df_t['correct_answer'] = df_t['correct_answer'].apply(str)
        df_t['response'] = df_t['response'].apply(str)
        for _, row in df_t.iterrows():
            formatted_example = get_formatted_training_example(row, tokenizer)
            records.append(formatted_example)

    random.shuffle(records)
    print("Num rows:", len(records))
    with open('../data/sft/train/moonshot_kimi_k2_data.jsonl', 'w') as outfile:
        for entry in records:
            json.dump(entry, outfile)
            outfile.write('\n')