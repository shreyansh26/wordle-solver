import re
import pandas as pd
from prompts import *
from openai import OpenAI
from transformers import AutoTokenizer

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:9200/v1"

client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def check_answer_format(answer):
    pattern = "<guess>(.*?)</guess>"
    match = re.search(pattern, answer)
    if match:
        return match.group(1).strip()
    else:
        return None

def get_feedback(model_answer, correct_answer):
    model_answer_chars = list(model_answer)
    correct_answer_chars = list(correct_answer)
    feedback_str = ""
    for i in range(len(model_answer_chars)):
        if model_answer_chars[i] == correct_answer_chars[i]:
            feedback_str += model_answer_chars[i] + "(✓) "
        elif model_answer_chars[i] in correct_answer_chars:
            feedback_str += model_answer_chars[i] + "(-) "
        else:
            feedback_str += model_answer_chars[i] + "(x) "
    return feedback_str.strip()

def check_answer(turn, model_answer, correct_answer):
    model_answer_extracted = check_answer_format(model_answer)
    # Break entire loop after 3 retries if this happens
    if model_answer_extracted is None:
        return False, "Model answer is not in the correct format"
    
    final_feedback_str = f"Guess {turn}: {model_answer_extracted} -> FEEDBACK: {get_feedback(model_answer_extracted, correct_answer)}"
    if model_answer_extracted == correct_answer:
        return True, final_feedback_str, model_answer_extracted
    else:
        return False, final_feedback_str, model_answer_extracted

def get_messages(turn=1, messages=None, past_response=None, last_turn_feedback=None):
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
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    return messages, prompt

def get_response(prompt):
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "min_p": 0}
    )
    return completion.choices[0].text

if __name__ == "__main__":
    word_list = open('../data/word_list.txt').read().splitlines()

    correct_answer = "APPLE"

    messages = None
    curr_response = None
    last_turn_feedback = None
    model_answer_extracted = None
    
    for turn in range(1, 7):
        if turn == 1:
            messages, prompt = get_messages(turn=turn)
        else:
            messages, prompt = get_messages(turn=turn, messages=messages, past_response=past_response, last_turn_feedback=last_turn_feedback)
        curr_response = get_response(prompt)
        is_correct, final_feedback_str, model_answer_extracted = check_answer(turn, curr_response, correct_answer)
        
        print("Model prediction: ", model_answer_extracted)
        print("Final feedback: ", final_feedback_str)

        if is_correct:
            print("Yay")
            break

        past_response = curr_response
    else:
        print(f"Answer not found. Final word: {model_answer_extracted}. Correct answer: {correct_answer}")