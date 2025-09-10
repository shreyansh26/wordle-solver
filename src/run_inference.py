import os
import pandas as pd
from openai import AsyncOpenAI
from create_sft_data import execute_turns, SamplingParams
import asyncio
import random
from logging_utils import get_logger
from transformers import AutoTokenizer

random.seed(1337)
DATASET_DIR = "../data/sft/moonshotai_Kimi-K2-Instruct"
# DATASET_DIR = "../data/sft/openai_gpt-oss-120b"
# MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen3/exp_2025-08-02T18:35:41_qwen3_4b_fsdp_packing=ffd_flash_attn_fsdp2_torch_compile_dcp_kimi_k2/epoch_5/step_final"
# MODEL_NAME = "/mnt/ssd2/shreyansh/models/qwen3/grpo_vllm_rl_v7/checkpoint-686"
MODEL_NAME = "/mnt/ssd2/shreyansh/models/llama32/exp_2025-09-09T00:14:29_llama32_3b_flash_attn_fsdp2_cp_torch_compile_dcp_deepseek_r1_sft/epoch_5/step_final"

logger = get_logger(f"inference_{MODEL_NAME.replace('/', '_')}")

def get_words_not_used():
    files = os.listdir(DATASET_DIR)
    words = []

    for file in files:
        if 'wordle_data' not in file:
            continue
        words.append(file.split('wordle_data_')[1].split('.csv')[0].lower())

    complete_words_list = open('../data/word_list.txt', 'r').read().splitlines()
    words_not_used = []

    words_not_used = list(set(complete_words_list) - set(words))

    # df_summary = pd.read_csv('../data/sft/train/moonshot_kimi_k2_summary.csv')
    df_summary = pd.read_csv('../data/sft/train/openai_gpt_oss-120b_summary.csv')

    words_failing = df_summary[df_summary['is_successful'] == "FAIL"].word.unique().tolist()

    return words_not_used, words_failing

def get_words_to_test():
    used_words_list = open('../data/word_list.txt', 'r').read().splitlines()
    complete_word_list = open('../data/valid_wordle_words.txt', 'r').read().splitlines()

    words_to_test = sorted(list(set(complete_word_list) - set(used_words_list)))
    random.shuffle(words_to_test)

    return words_to_test[:200]


async def process_word_chunk(sample_word_list, model_name, tokenizer, verbose=False, client=None, sampling_params=None, model_data_dir=None):
    """Process a chunk of words in parallel"""
    logger.info(f"Processing chunk of {len(sample_word_list)} words: {sample_word_list}")
    tasks = [execute_turns(word, model_name, tokenizer=tokenizer, verbose=verbose, response_provider="local", client=client, sampling_params=sampling_params, model_data_dir=model_data_dir) for word in sample_word_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_word_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception for word {sample_word_list[i]}: {result}")
        elif result is not None:
            successful_word_results.append(result)
    
    logger.info(f"Successfully processed {len(successful_word_results)} out of {len(sample_word_list)} words in chunk")
    return successful_word_results

async def main():
    # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0)
    sampling_params = SamplingParams(temperature=0.2, top_p=1, top_k=-1, min_p=0)
    client = AsyncOpenAI(
                api_key="EMPTY",
                base_url="http://localhost:9203/v1",
            )
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    model_data_dir = os.path.join("..", "data", "sft", MODEL_NAME.replace("/", "_"))
    os.makedirs(model_data_dir, exist_ok=True)

    # words_not_used, words_failing = get_words_not_used()
    # print("Words not used:", len(words_not_used))
    # print("Words failing:", len(words_failing))

    words_to_test = get_words_to_test()

    sample_word_list = words_to_test
    chunk_size = 5

    all_successful_words = []
    retry_count_list = []
    turn_count_list = []
    success_failure_list = []

    for i in range(0, len(sample_word_list), chunk_size):
        chunk = sample_word_list[i:i + chunk_size]
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(sample_word_list) + chunk_size - 1)//chunk_size}")
        # chunk = [word.upper() if random.random() < 0.5 else word.lower() for word in chunk]
        successful_words_and_retry_count = await process_word_chunk(chunk, MODEL_NAME, tokenizer=tokenizer, verbose=False, client=client, sampling_params=sampling_params, model_data_dir=model_data_dir)
        successful_words = [x[0] for x in successful_words_and_retry_count]
        retry_count = [x[1] for x in successful_words_and_retry_count]
        turn_count = [x[2] for x in successful_words_and_retry_count]
        success_failure = [x[3] for x in successful_words_and_retry_count]
        all_successful_words.extend(successful_words)
        retry_count_list.extend(retry_count)
        turn_count_list.extend(turn_count)
        success_failure_list.extend(success_failure)

    logger.info(f"Total processing complete. Successfully processed {len(all_successful_words)} out of {len(sample_word_list)} words")
    summary_df = pd.DataFrame({
        'processed_words': all_successful_words,
        'model_name': [MODEL_NAME] * len(all_successful_words),
        'retry_count': retry_count_list,
        'turn_count': turn_count_list,
        'success_failure': success_failure_list
    })
    summary_df.to_csv(f"{model_data_dir}/processing_summary.csv", index=False)
    logger.info(f"Summary saved to {model_data_dir}/processing_summary.csv")

if __name__ == "__main__":
    asyncio.run(main())