import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline
import os
import pandas as pd
import json
import re
import time

class Feature_extractor():
    def __init__(self, model="cahya/roberta-base-indonesian-522M", max_lenth=512, device=0):
        self.feature_extractor = pipeline('feature-extraction',model=model, device=device, tokenizer=model, truncation=True,padding='max_length', add_special_tokens = True, max_length=512)
        self.feature_extractor.tokenizer.model_max_length = max_lenth

    def feature_extract(self, response):
        '''response: [batch, text]'''
        response_tensors = []
        for text in response:
            if re.search(r'Tweet \d+:', text):
                segments = re.split(r'Tweet \d+:', text.strip().replace('\n', ' '))
                segments = [segments[i] for i in range(1, len(segments))]
            elif re.search(r'\d+\.', text):
                segments = re.split(r'\d+\.', text.strip().replace('\n', ' '))
                segments = [segments[i] for i in range(1, len(segments))]
            elif re.search(r'\d+:', text):
                segments = re.split(r'\d+:', text.strip().replace('\n', ' '))
                segments = [segments[i] for i in range(1, len(segments))]
            elif re.search(r'\n\n', text):
                segments = re.split(r'\n\n', text.strip())
                segments = [segments[i] for i in range(1, len(segments))]
            else:
                segments = [text]
            if len(segments) == 0:
                total_response_tensor = torch.zeros(768)
            else:
                all_tweet_tensor=torch.tensor(self.feature_extractor(segments, padding=True, max_length=512, truncation=True, batch_size=len(segments)))
                total_word_tensor = torch.mean(all_tweet_tensor, dim=2)
                total_response_tensor = torch.mean(total_word_tensor, dim=0)
            response_tensors.append(total_response_tensor.squeeze())
        response_tensors = torch.stack(response_tensors)
        return response_tensors