import re
import os
import json
import ast
import random
import pandas as pd
from transformers import AutoTokenizer

random.seed(1337)
MAX_LENGTH = 12288
DATASET_DIR = '../data/sft/moonshotai_Kimi-K2-Instruct'

def dump_to_jsonl(records, path):
    with open(path, 'w') as outfile:
        for entry in records:
            json.dump(entry, outfile)
            outfile.write('\n')

def perform_validation(formatted_input, formatted_output):
    if "<think>" not in formatted_output:
        print("No <think> in formatted_output")
        return False
    if "</think>" not in formatted_output:
        print("No </think> in formatted_output")
        return False
    if "<guess>" not in formatted_output:
        print("No <guess> in formatted_output")
        return False
    if "</guess>" not in formatted_output:
        print("No </guess> in formatted_output")
        return False
    
    predicted_think = re.compile("<think>(.*?)</think>", re.DOTALL)
    match_think = predicted_think.search(formatted_output)
    if match_think is None:
        print("Nothing between <think> and </think> in formatted_output")
        return False
    predicted = match_think.group(1).strip()
    if predicted is None:
        print("No predicted thinking in formatted_output")
        return False
    
    predicted_guess = re.compile("<guess>(.*?)</guess>", re.DOTALL)
    match_guess = predicted_guess.search(formatted_output)
    if match_guess is None:
        print("Nothing between <guess> and </guess> in formatted_output")
        return False
    predicted = match_guess.group(1).strip()
    if predicted is None:
        print("No predicted word in formatted_output")
        return False
    if len(predicted) != 5:
        print("Predicted word is not 5 characters")
        return False

    complete_input = f"{formatted_input}{formatted_output}"
    complete_input_len = len(tokenizer.encode(complete_input))

    if complete_input_len > MAX_LENGTH:
        return False

    return True

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

    if not perform_validation(formatted_input, formatted_output):
        print(f"Skipping word {word} because it's invalid")
        return None
    else:
        return {
            "instruction": formatted_input,
            "output": formatted_output,
            "word": word,
            "turn": turn,
        }

if __name__ == "__main__":
    df = pd.read_csv('../data/sft/train/moonshot_kimi_k2_summary.csv', dtype={'word': str, 'num_rows': int, 'is_successful': str})
    df = df[df['is_successful'] == "SUCCESS"]
    print("Successful words:", df.shape[0])

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    records = []
    train_records = []
    val_records = []

    num_unique_words = df['word'].nunique()
    num_unique_words_train = int(num_unique_words * 0.8)
    num_unique_words_val = num_unique_words - num_unique_words_train
    unique_words = df['word'].unique()
    random.shuffle(unique_words)
    print("Num unique words:", num_unique_words)
    print("Num unique words train:", num_unique_words_train)
    print("Num unique words val:", num_unique_words_val)

    print(f"Using {num_unique_words_train} words for train and {num_unique_words_val} words for val")
    for word in unique_words[:num_unique_words_train]:
        df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
        df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
        df_t['correct_answer'] = df_t['correct_answer'].apply(str)
        df_t['response'] = df_t['response'].apply(str)
        for _, row in df_t.iterrows():
            formatted_example = get_formatted_training_example(row, tokenizer)
            if formatted_example is not None:   
                records.append(formatted_example)
                train_records.append(formatted_example)

    for word in unique_words[num_unique_words_train:]:
        df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
        df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
        df_t['correct_answer'] = df_t['correct_answer'].apply(str)
        df_t['response'] = df_t['response'].apply(str)
        for _, row in df_t.iterrows():
            formatted_example = get_formatted_training_example(row, tokenizer)
            if formatted_example is not None:   
                records.append(formatted_example)
                val_records.append(formatted_example)

    random.shuffle(records)
    random.shuffle(train_records)
    random.shuffle(val_records)

    print("Num rows:", len(records))
    dump_to_jsonl(records, '../data/sft/train/moonshot_kimi_k2_data.jsonl')

    rows_train = train_records
    print("Num rows train:", len(rows_train))
    dump_to_jsonl(rows_train, '../data/sft/train/moonshot_kimi_k2_data_train.jsonl')

    rows_val = val_records
    print("Num rows val:", len(rows_val))
    dump_to_jsonl(rows_val, '../data/sft/train/moonshot_kimi_k2_data_val.jsonl')