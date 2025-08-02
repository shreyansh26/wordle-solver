import os
import re
import asyncio
import random
import pandas as pd
from prompts import *
from openai import AsyncOpenAI
from api_key import TOGETHER_API_KEY, FIREWORKS_API_KEY
from transformers import AutoTokenizer
from dotenv import load_dotenv
from logging_utils import get_logger
from pytz import timezone 
from datetime import datetime

# Load HF_HOME
load_dotenv()

response_provider = "together"

model_name = "moonshotai/Kimi-K2-Instruct"
# model_name = "deepseek-ai/DeepSeek-R1-0528"
# model_name = "Qwen/Qwen3-235B-A22B-fp8-tput"
# model_name = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"
# model_name = "accounts/fireworks/models/glm-4p5"

if response_provider == "together":
    hf_model_name = model_name if "qwen3" not in model_name.lower() else "Qwen/Qwen3-235B-A22B"
elif response_provider == "fireworks":
    # Set manually for any run
    hf_model_name = "Qwen/Qwen3-235B-A22B"
else:
    hf_model_name = model_name

# Create directory for model data
model_data_dir = os.path.join("..", "data", "sft", model_name.replace("/", "_"))
logger = get_logger(f"create_sft_data_{model_name.replace('/', '_')}")
os.makedirs(model_data_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)

class SamplingParams:
    def __init__(self, temperature=0.6, top_p=0.95, top_k=20, min_p=0):
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

def check_answer_format(answer):
    pattern_guess = re.compile("<guess>(.*?)</guess>", re.DOTALL)
    match_guess = pattern_guess.search(answer)
    pattern_think = re.compile("<think>(.*?)</think>", re.DOTALL)
    match_think = pattern_think.search(answer)
    if match_guess is None or match_think is None:
        return None, None
    else:
        thinking = match_think.group(1).strip()
        guess = match_guess.group(1).strip()

        # Sometime R1 adds more text in between </think> and <guess> i.e. end of thinking and start of guess
        # Include that in the think text
        pattern_additional_text = re.compile("</think>(.*?)<guess>", re.DOTALL)
        match_additional_text = pattern_additional_text.search(answer)
        additional_text = match_additional_text.group(1).strip() if match_additional_text else ""
        return guess, thinking + additional_text

def get_feedback(model_answer, correct_answer):
    model_answer_chars = list(map(lambda x: x.lower(), model_answer))
    correct_answer_chars = list(map(lambda x: x.lower(), correct_answer))
    feedback_str = ""
    if len(model_answer_chars) != len(correct_answer_chars):
        logger.info(f"Model answer is not five letters. Model answer: {model_answer}, Correct answer: {correct_answer}")
        return "Model answer is not five letters."
    for i in range(len(model_answer_chars)):
        if model_answer_chars[i] == correct_answer_chars[i]:
            feedback_str += model_answer_chars[i].upper() + "(✓) "
        elif model_answer_chars[i] in correct_answer_chars:
            feedback_str += model_answer_chars[i].upper() + "(-) "
        else:
            feedback_str += model_answer_chars[i].upper() + "(x) "
    return feedback_str.strip()

def check_answer(turn, model_answer, correct_answer):
    model_answer_extracted, model_think_extracted = check_answer_format(model_answer)

    if model_answer_extracted is None or model_think_extracted is None:
        return False, "Model answer is not in the correct format", "Incorrect format answer", "Incorrect format think"
    
    final_feedback_str = f"Guess {turn}: {model_answer_extracted} -> FEEDBACK: {get_feedback(model_answer_extracted, correct_answer)}"
    if model_answer_extracted.lower() == correct_answer.lower():
        return True, final_feedback_str, model_answer_extracted, model_think_extracted
    else:
        return False, final_feedback_str, model_answer_extracted, model_think_extracted

def get_messages(turn=1, messages=None, past_response=None, last_turn_feedback=None, tokenizer=None):
    assert (turn == 1 and messages is None) or (turn > 1 and messages is not None)
    if turn == 1:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_FIRST}
        ]
    else:
        messages += [
            {"role": "assistant", "content": past_response},
            {"role": "user", "content": USER_PROMPT_SUBSEQUENT.format(feedback_str=last_turn_feedback)}
        ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return messages, prompt

async def get_response(prompt, messages, model_name, client=None, sampling_params=None):
    completion = await client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=8192,
        temperature=sampling_params.temperature if sampling_params is not None else 0.6,
        top_p=sampling_params.top_p if sampling_params is not None else 0.95,
        extra_body={"top_k": sampling_params.top_k if sampling_params is not None else 20, "min_p": sampling_params.min_p if sampling_params is not None else 0}
    )
    return completion.choices[0].text

async def get_response_together(prompt, messages, model_name, client=None, sampling_params=None):
    completion = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192,
        stream=False
    )
    return completion.choices[0].message.content

async def execute_turns(correct_answer, model_name, tokenizer=None, verbose=False, response_provider="together", client=None, sampling_params=None, model_data_dir=None):
    if response_provider == "together":
        get_response_fn = get_response_together
    elif response_provider == "fireworks":
        get_response_fn = get_response_together
    elif response_provider == "local":
        get_response_fn = get_response
    else:
        raise ValueError(f"Invalid response provider: {response_provider}")

    try:
        messages = None
        curr_response = None
        last_turn_feedback = ""
        model_answer_extracted = None
        final_answer = "FAIL"
        rows = []
        total_retries = 0
        
        # Prepare messages for the first turn
        messages, prompt = get_messages(turn=1, tokenizer=tokenizer)

        for turn in range(1, 7):
            if verbose:
                print(f"Turn {turn}: {prompt}")
                print("-"*100)

            curr_response = await get_response_fn(prompt, messages, model_name, client=client, sampling_params=sampling_params)
            is_correct, final_feedback_str, model_answer_extracted, model_think_extracted = check_answer(turn, curr_response, correct_answer)

            if verbose:
                print(f"Response: {curr_response}")
                print("="*50)
                print(f"Feedback: {final_feedback_str}")
                print("="*50)
                print("-"*100)

            cnt_retries = 0
            while final_feedback_str == "Model answer is not in the correct format":
                logger.info(f"Model answer is not in the correct format for {correct_answer}. Retrying... {cnt_retries}")
                cnt_retries += 1
                if cnt_retries > 3:
                    break
                curr_response = await get_response_fn(prompt, messages, model_name, client=client, sampling_params=sampling_params)
                is_correct, final_feedback_str, model_answer_extracted, model_think_extracted = check_answer(turn, curr_response, correct_answer)
                if verbose:
                    print(f"Response: {curr_response}")
                    print("="*50)
                    print(f"Feedback: {final_feedback_str}")
                    print("="*50)
                    print("-"*100)

            total_retries += cnt_retries
            
            if is_correct:
                final_answer = "SUCCESS"
                logger.info(f"Answer found in turn {turn} for {correct_answer}. Final word: {model_answer_extracted}")

            # Store all data for the current turn
            curr_row = {
                "model_name": model_name,
                "turn": turn,
                "correct_answer": correct_answer,
                "prompt": prompt,
                "messages": messages.copy(),
                "response": curr_response,
                "model_answer": model_answer_extracted,
                "model_think": model_think_extracted,
                "feedback": final_feedback_str,
                "curr_answer": final_answer,
            }
            rows.append(curr_row)

            if is_correct:
                break

            # If not correct and not the last turn, prepare for the next turn
            if turn < 6:
                past_response = model_think_extracted + "\n\n" + "<guess>" + model_answer_extracted + "</guess>"
                last_turn_feedback += "\n" + final_feedback_str
                messages, prompt = get_messages(turn=turn + 1, messages=messages, past_response=past_response, last_turn_feedback=last_turn_feedback, tokenizer=tokenizer)

        else:
            logger.info(f"Answer not found for {correct_answer}. Final word: {model_answer_extracted}")
            final_answer = "FAIL"

        df_wordle_data = pd.DataFrame(rows)

        df_wordle_data.to_csv(f"{model_data_dir}/wordle_data_{correct_answer}.csv", index=False)
        logger.info(f"Successfully processed word: {correct_answer}")
        return correct_answer, total_retries, turn, final_answer
    
    except Exception as e:
        if "InternalServerError" in str(type(e)) or "InternalServerError" in str(e):
            logger.warning(f"InternalServerError encountered for word {correct_answer}. Skipping...")
        else:
            logger.error(f"Error processing word {correct_answer}: {e}")
        return correct_answer, total_retries, turn, f"FAIL;{e}"

async def process_word_chunk(words_chunk, model_name, tokenizer=None, verbose=False, client=None, sampling_params=None, model_data_dir=None):
    """Process a chunk of words in parallel"""
    logger.info(f"Processing chunk of {len(words_chunk)} words: {words_chunk}")
    tasks = [execute_turns(word, model_name, tokenizer=tokenizer, verbose=verbose, response_provider="together", client=client, sampling_params=sampling_params, model_data_dir=model_data_dir) for word in words_chunk]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_word_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception for word {words_chunk[i]}: {result}")
        elif result is not None:
            successful_word_results.append(result)
    
    logger.info(f"Successfully processed {len(successful_word_results)} out of {len(words_chunk)} words in chunk")
    return successful_word_results

async def main():
    curr_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M:%S')
    processed_words = []
    
    for file in os.listdir(model_data_dir):
        if "wordle_data" in file:
            word = file.split('wordle_data_')[1].split('.csv')[0]
            processed_words.append(word.lower())
    
    processed_words = list(set(processed_words))
    
    if response_provider == "together":
        client = AsyncOpenAI(
                api_key=TOGETHER_API_KEY,
                base_url="https://api.together.xyz/v1",
            )
    elif response_provider == "fireworks":
        client = AsyncOpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
            )
    else:
        client = AsyncOpenAI(
                api_key="EMPTY",
                base_url="http://localhost:9200/v1",
            )
    if response_provider == "local":
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0)
    else:
        sampling_params = None
    
    # Load word list
    with open('../data/word_list.txt', 'r') as f:
        word_list = f.read().splitlines()
    
    # Sample words randomly
    # sample_words = random.sample(word_list, min(1000, len(word_list)))
    sample_words = sorted(list(set(word_list) - set(processed_words)))
    logger.info(f"Selected {len(sample_words)} words to process")
    
    chunk_size = 20
    all_successful_words = []
    retry_count_list = []
    turn_count_list = []
    success_failure_list = []
    
    for i in range(0, len(sample_words), chunk_size):
        chunk = sample_words[i:i + chunk_size]
        logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(sample_words) + chunk_size - 1)//chunk_size}")
        words_chunk_randomcase = [word.upper() if random.random() < 0.5 else word.lower() for word in chunk]
        successful_words_and_retry_count = await process_word_chunk(words_chunk_randomcase, model_name, tokenizer=tokenizer, client=client, sampling_params=sampling_params, model_data_dir=model_data_dir)
        successful_words = [x[0] for x in successful_words_and_retry_count]
        retry_count = [x[1] for x in successful_words_and_retry_count]
        turn_count = [x[2] for x in successful_words_and_retry_count]
        success_failure = [x[3] for x in successful_words_and_retry_count]
        all_successful_words.extend(successful_words)
        retry_count_list.extend(retry_count)
        turn_count_list.extend(turn_count)
        success_failure_list.extend(success_failure)
        
        # Small delay between chunks to avoid overwhelming the API
        await asyncio.sleep(1)
    
    logger.info(f"Total processing complete. Successfully processed {len(all_successful_words)} out of {len(sample_words)} words")
    
    # Save summary
    summary_df = pd.DataFrame({
        'processed_words': all_successful_words,
        'model_name': [model_name] * len(all_successful_words),
        'retry_count': retry_count_list,
        'turn_count': turn_count_list,
        'success_failure': success_failure_list
    })
    summary_df.to_csv(f"{model_data_dir}/processing_summary_{curr_time}.csv", index=False)
    logger.info(f"Summary saved to {model_data_dir}/processing_summary_{curr_time}.csv")

if __name__ == "__main__":
    asyncio.run(main())