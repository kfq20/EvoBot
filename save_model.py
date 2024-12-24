from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

model_name = "/home/fanqi/llm_simulation/TwiBot-22/src/BotRGCN/datasets/Twibot-22/sft_data"
llm = LLM("NousResearch/Llama-2-7b-chat-hf", tensor_parallel_size=1, dtype='float16', max_lora_rank=64, gpu_memory_utilization=0.7, enable_lora=True)
sampling_params = SamplingParams(temperature=1.0, max_tokens=512, repetition_penalty=1.3)
prompts = "hello"

outputs = llm.generate(prompts,
                       sampling_params,
                       lora_request=LoRARequest("sft", 1, f"{model_name}"))

for output in outputs:
    print(output.outputs[0].text)

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)
# model_path = "sft_model"
# model.save_pretrained(model_path)