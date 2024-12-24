import openai
import os

class GPT4Simulator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_text(self, prompt, max_tokens=50):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=1,
        )
        return response['choices'][0]['message']['content']