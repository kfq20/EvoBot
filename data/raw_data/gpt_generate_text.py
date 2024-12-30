import os
from typing import List, Dict
from openai import AsyncClient
from openai import AzureOpenAI
from omegaconf import OmegaConf

REGION = "canadaeast"
MODEL = "gpt-35-turbo-0125"
API_KEY = "f974444c22879c2a3e0249e34d561539"

API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
ENDPOINT = f"{API_BASE}/{REGION}"

class AsyncChatLLM:
    """
    Wrapper for an (Async) Chat Model.
    """
    def __init__(
        self, 
        kwargs: Dict[str, str],         
        ):
        """
        Initializes AsynceOpenAI client.
        """
        self.model = kwargs.pop("model")
        if self.model == "gpt-4-1106-preview" or self.model == "gpt-4o" or self.model == "gpt-3.5-turbo-1106":
            pass
        else:
            #OmegaConf.set_struct(kwargs, False) 
            base_url = kwargs.pop("base_url")
            port = kwargs.pop("port")
            version = kwargs.pop("version")
            kwargs["base_url"] = f"{base_url}:{port}/{version}"            
            #OmegaConf.set_struct(kwargs, True)
        
        # self.client = AsyncClient(**kwargs)
        self.client = AzureOpenAI(
            api_key=API_KEY,
            api_version="2024-02-01",
            azure_endpoint=ENDPOINT,
        )

    @property
    def llm_type(self):
        return "AsyncClient"

    async def __call__(self, 
        messages: List[Dict[str, str]], 
        **kwargs,
    ):
        """
        Make an async API call.
        """        
        # Mixtral has to follow a different format: ['system', 'assistant', 'user', ...]
        if self.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            user_message = messages.pop()
            assistant_message = messages.pop()
            assistant_message["role"] = "assistant"
            messages.append(user_message)
            messages.append(assistant_message)

        return await self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            )