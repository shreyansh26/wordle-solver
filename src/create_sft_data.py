import re
import pandas as pd
from prompts import *
from openai import OpenAI
from api_key import TOGETHER_API_KEY
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

USE_TOGETHER_MODELS = True

if USE_TOGETHER_MODELS:
    client = OpenAI(
            api_key=TOGETHER_API_KEY,
            base_url="https://api.together.xyz/v1",
        )
else:
    client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:9200/v1",
        )

model_name = "moonshotai/Kimi-K2-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def check_answer_format(answer):
    pattern_guess = re.compile("<guess>(.*?)</guess>", re.DOTALL)
    match_guess = pattern_guess.search(answer)
    pattern_think = re.compile("<think>(.*?)</think>", re.DOTALL)
    match_think = pattern_think.search(answer)
    if match_guess is None or match_think is None:
        return None, None
    else:
        return match_guess.group(1).strip(), match_think.group(1).strip()

def get_feedback(model_answer, correct_answer):
    model_answer_chars = list(map(lambda x: x.lower(), model_answer))
    correct_answer_chars = list(map(lambda x: x.lower(), correct_answer))
    feedback_str = ""
    for i in range(len(model_answer_chars)):
        if model_answer_chars[i] == correct_answer_chars[i]:
            feedback_str += model_answer_chars[i].upper() + "(✓) "
        elif model_answer_chars[i] in correct_answer_chars:
            feedback_str += model_answer_chars[i].upper() + "(-) "
        else:
            feedback_str += model_answer_chars[i].upper() + "(x) "
    return feedback_str.strip()

def check_answer(turn, model_answer, correct_answer):
    # print(model_answer)
    model_answer_extracted, model_think_extracted = check_answer_format(model_answer)
    # Break entire loop after 3 retries if this happens
    if model_answer_extracted is None or model_think_extracted is None:
        return False, "Model answer is not in the correct format", "Incorrect format answer", "Incorrect format think"
    
    final_feedback_str = f"Guess {turn}: {model_answer_extracted} -> FEEDBACK: {get_feedback(model_answer_extracted, correct_answer)}"
    if model_answer_extracted == correct_answer:
        return True, final_feedback_str, model_answer_extracted, model_think_extracted
    else:
        return False, final_feedback_str, model_answer_extracted, model_think_extracted

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

def get_response(prompt, messages):
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "min_p": 0}
    )
    return completion.choices[0].text

def get_response_together(prompt, messages):
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=8192,
        stream=False
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    word_list = open('../data/word_list.txt').read().splitlines()

    correct_answer = "APPLE"

    messages = None
    curr_response = None
    last_turn_feedback = ""
    model_answer_extracted = None
    
    for turn in range(1, 7):
        if turn == 1:
            messages, prompt = get_messages(turn=turn)
        else:
            messages, prompt = get_messages(turn=turn, messages=messages, past_response=past_response, last_turn_feedback=last_turn_feedback)

        print(prompt)
        print("-"*50)
        curr_response = get_response_together(prompt, messages)
        print(curr_response)
        print("="*100)

        is_correct, final_feedback_str, model_answer_extracted, model_think_extracted = check_answer(turn, curr_response, correct_answer)
        
        print("Model prediction: ", model_answer_extracted)
        print("Final feedback: ", final_feedback_str)

        if is_correct:
            print("Yay")
            break

        past_response = model_think_extracted + "\n\n" + "<guess>" + model_answer_extracted + "</guess>"
        last_turn_feedback += "\n" + final_feedback_str
        print("*"*100)
    else:
        print(f"Answer not found. Final word: {model_answer_extracted}. Correct answer: {correct_answer}")