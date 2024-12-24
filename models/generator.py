import torch.nn as nn
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from vllm import LLM, SamplingParams
import re

class Generator(nn.Module):
    def __init__(self, model_path, **kwargs):
        super(Generator, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        # self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=2048)
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

    def generate_text(self, prompt, max_length=50, temperature=1.0, do_sample=False, repetition_penalty=1.0):
        inputs = self.tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=2048).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_length//2, temperature=temperature, do_sample=do_sample, repetition_penalty=repetition_penalty, pad_token_id = self.tokenizer.pad_token_id)
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return self.post_process(output_text)
    
    def post_process(self, outputs):
        processed_outputs = []
        for output in outputs:
            response_output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
            # if re.search(r'Tweet \d+:', response_output):
            #     segments = re.split(r'Tweet \d+:', response_output.strip().replace('\n', ' '))
            #     segments = [segments[i] for i in range(1, len(segments))]
            # elif re.search(r'\d+\.', response_output):
            #     segments = re.split(r'\d+\.', response_output.strip().replace('\n', ' '))
            #     segments = [segments[i] for i in range(1, len(segments))]
            # elif re.search(r'\d+:', response_output):
            #     segments = re.split(r'\d+:', response_output.strip().replace('\n', ' '))
            #     segments = [segments[i] for i in range(1, len(segments))]
            # elif re.search(r'\n\n', response_output):
            #     segments = re.split(r'\n\n', response_output.strip())
            #     segments = [segments[i] for i in range(1, len(segments))]
            # else:
            #     segments = [response_output]
            processed_outputs.append(response_output)

        return processed_outputs
    
    def update_model(self, model):
        self.model = model