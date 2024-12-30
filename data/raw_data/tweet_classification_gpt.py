import os
from typing import List, Dict
from openai import AsyncClient
from openai import AzureOpenAI
from omegaconf import OmegaConf

REGION = "northcentralus"
MODEL = "gpt-35-turbo-0125"
API_KEY = "f974444c22879c2a3e0249e34d561539"

API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
ENDPOINT = f"{API_BASE}/{REGION}"

client = AzureOpenAI(
            api_key=API_KEY,
            api_version="2024-02-01",
            azure_endpoint=ENDPOINT,
        )

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Say Hello!"}
    ],
)

# print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)