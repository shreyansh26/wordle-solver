import re
import os
import json
import ast
import random
import pandas as pd
from transformers import AutoTokenizer

random.seed(1337)
MAX_LENGTH = 16384 # 12288
# DATASET_DIR = '../data/sft/moonshotai_Kimi-K2-Instruct'
# DATASET_DIR = '../data/sft/deepseek-ai_DeepSeek-R1-0528'
DATASET_DIR = '../data/sft/openai_gpt-oss-120b'

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
    # df = pd.read_csv('../data/sft/train/moonshot_kimi_k2_summary_v2.csv', dtype={'word': str, 'num_rows': int, 'is_successful': str})
    # df = pd.read_csv('../data/sft/train/deepseek_r1_summary.csv', dtype={'word': str, 'num_rows': int, 'is_successful': str})
    df = pd.read_csv('../data/sft/train/openai_gpt_oss-120b_summary.csv', dtype={'word': str, 'num_rows': int, 'is_successful': str})
    df = df[df['is_successful'] == "SUCCESS"]
    print("Successful words:", df.shape[0])

    df['low_word'] = df['word'].str.lower()
    df = df.sort_values(by='num_rows', ascending=False)
    df = df.drop_duplicates(subset=['low_word'])
    df = df.drop(columns=['low_word'])

    print("Unique successful words:", df.shape[0])

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    DO_RL = True

    if DO_RL:
        SFT_PERC = 0.6
    else:
        SFT_PERC = 1

    records = []
    train_records_sft = []
    val_records_sft = []

    train_records_rl = []
    val_records_rl = []

    num_unique_words = df['word'].nunique()

    if DO_RL:
        num_unique_words_sft = int(num_unique_words * SFT_PERC)
        num_unique_words_rl = num_unique_words - num_unique_words_sft
    else:
        num_unique_words_sft = int(num_unique_words * SFT_PERC)
        num_unique_words_rl = 0

    unique_words = df['word'].unique()
    random.shuffle(unique_words)
    print("Num unique words:", num_unique_words)
    print("Num unique words sft:", num_unique_words_sft)
    print("Num unique words rl:", num_unique_words_rl)

    print(f"Using {num_unique_words_sft} words for sft and {num_unique_words_rl} words for rl")

    sft_words = unique_words[:num_unique_words_sft]
    rl_words = unique_words[num_unique_words_sft:]

    # SFT Train and Val split
    num_unique_words_sft_train = int(num_unique_words_sft * 0.8)
    num_unique_words_sft_val = num_unique_words_sft - num_unique_words_sft_train

    sft_words_train = sft_words[:num_unique_words_sft_train]
    sft_words_val = sft_words[num_unique_words_sft_train:]

    for word in sft_words_train:
        df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
        df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
        df_t['correct_answer'] = df_t['correct_answer'].apply(str)
        df_t['response'] = df_t['response'].apply(str)
        for _, row in df_t.iterrows():
            formatted_example = get_formatted_training_example(row, tokenizer)
            if formatted_example is not None:   
                records.append(formatted_example)
                train_records_sft.append(formatted_example)

    for word in sft_words_val:
        df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
        df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
        df_t['correct_answer'] = df_t['correct_answer'].apply(str)
        df_t['response'] = df_t['response'].apply(str)
        for _, row in df_t.iterrows():
            formatted_example = get_formatted_training_example(row, tokenizer)
            if formatted_example is not None:   
                records.append(formatted_example)
                val_records_sft.append(formatted_example)

    # RL Train and Val split
    if DO_RL:
        num_unique_words_rl_train = int(num_unique_words_rl * 0.8)
        num_unique_words_rl_val = num_unique_words_rl - num_unique_words_rl_train

        rl_words_train = rl_words[:num_unique_words_rl_train]
        rl_words_val = rl_words[num_unique_words_rl_train:]

        for word in rl_words_train:
            df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
            df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
            df_t['correct_answer'] = df_t['correct_answer'].apply(str)
            df_t['response'] = df_t['response'].apply(str)
            for _, row in df_t.iterrows():
                formatted_example = get_formatted_training_example(row, tokenizer)
                if formatted_example is not None:   
                    records.append(formatted_example)
                    train_records_rl.append(formatted_example)

        for word in rl_words_val:
            df_t = pd.read_csv(os.path.join(DATASET_DIR, f'wordle_data_{word}.csv'))
            df_t['messages'] = df_t['messages'].apply(ast.literal_eval)
            df_t['correct_answer'] = df_t['correct_answer'].apply(str)
            df_t['response'] = df_t['response'].apply(str)
            for _, row in df_t.iterrows():
                formatted_example = get_formatted_training_example(row, tokenizer)
                if formatted_example is not None:   
                    records.append(formatted_example)
                    val_records_rl.append(formatted_example)

    random.shuffle(train_records_sft)
    random.shuffle(val_records_sft)

    random.shuffle(train_records_rl)
    random.shuffle(val_records_rl)

    print("Num rows:", len(records))
    # dump_to_jsonl(records, '../data/sft/train/moonshot_kimi_k2_data_v2.jsonl')
    # dump_to_jsonl(records, '../data/sft/train/deepseek_r1_data.jsonl')
    dump_to_jsonl(train_records_rl, '../data/sft/train/openai_gpt_oss-120b_data.jsonl')

    print("Num rows train (sft):", len(train_records_sft))
    # dump_to_jsonl(rows_train, '../data/sft/train/moonshot_kimi_k2_data_train_v2.jsonl')
    # dump_to_jsonl(rows_train, '../data/sft/train/deepseek_r1_data_train.jsonl')
    dump_to_jsonl(train_records_rl, '../data/sft/train/openai_gpt_oss-120b_data_sft_train.jsonl')

    print("Num rows val (sft):", len(val_records_sft))
    # dump_to_jsonl(rows_val, '../data/sft/train/moonshot_kimi_k2_data_val_v2.jsonl')
    # dump_to_jsonl(rows_val, '../data/sft/train/deepseek_r1_data_val.jsonl')
    dump_to_jsonl(train_records_rl, '../data/sft/train/openai_gpt_oss-120b_data_sft_val.jsonl')

    if DO_RL:
        print("Num rows train (rl):", len(train_records_rl))
        dump_to_jsonl(train_records_rl, '../data/sft/train/openai_gpt_oss-120b_data_rl_train.jsonl')

        print("Num rows val (rl):", len(val_records_rl))
        dump_to_jsonl(val_records_rl, '../data/sft/train/openai_gpt_oss-120b_data_rl_val.jsonl')