import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel,AutoPeftModelForCausalLM

def merge_peft_adapters(adapter_dir, output_path):
    torch.cuda.empty_cache()
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_dir, device_map="cpu", torch_dtype=torch.float16)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    output_merged_dir = os.path.join(output_path)
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_merged_dir)

if __name__ == '__main__':
    merge_peft_adapters(adapter_dir=f"./models/SFT/result_comm_0", output_path=f"./models/SFT/sft_merged_ckp_0")