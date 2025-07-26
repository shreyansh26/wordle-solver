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

def create_prompt():
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_FIRST

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    # )
    # print(text)

    # return text
    return messages


if __name__ == "__main__":
    prompt = create_prompt()
    print(prompt)
    # completion = client.completions.create(
    #     model=model_name,
    #     prompt=prompt,
    #     max_tokens=8192,
    #     temperature=0.6,
    #     top_p=0.95,
    #     extra_body={"top_k": 20, "min_p": 0}
    # )
    # print(completion.choices[0].text)
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        max_tokens=8192,
        temperature=0.6,
        top_p=0.95,
        extra_body={"top_k": 20, "min_p": 0}
    )
    print(response.choices[0])
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    print("reasoning_content:", reasoning_content)
    print("content:", content)
    