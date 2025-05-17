import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, DPOTrainer
from peft import LoraConfig
from torch.utils.data import Dataset, DataLoader
from utils.sft_data_generation import instruction_tune_instance
from utils.merge_peft_adapters import merge_peft_adapters
import json
import torch
import os
os.environ["WANDB_DISABLED"]="true"

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--input", default="result", help="sft data name, those in corpus/")
    argParser.add_argument("--model", default="llama2_7b", help="model name, e.g. mistral")
    argParser.add_argument("--parent_directory", default="./models/SFT", help="parent directory") # other_checkpoint/
    argParser.add_argument("--epochs", default=5, help="number of epochs")
    argParser.add_argument("--comm", default=0, help="community index")
    argParser.add_argument("--data-num", default=1024, help="sft data num")

    args = argParser.parse_args()
    folder_path = f"./data/raw_data/community_{args.comm}"
    processed_path = f"./data/processed_data/community_{args.comm}"
    output_dir = f"./data/sft_data/community_{args.comm}"
    if not os.path.exists(output_dir+"/sft_data.jsonl"):
        instruction_tune_instance(folder_path, processed_path, output_dir, args.data_num)
    input = args.input
    model_name = args.model
    parent_directory = args.parent_directory
    epochs = int(args.epochs)

    dataset = load_dataset("json", data_files=f"{output_dir}/sft_data.jsonl", split="train")

    if model_name == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif model_name == "llama2_7b":
        model_name = "NousResearch/Llama-2-7b-chat-hf"
    elif model_name == "llama2_13b":
        model_name = "meta-llama/Llama-2-13b-chat-hf"
    elif model_name == "llama2_70b":
        model_name = "meta-llama/Llama-2-70b-chat-hf"

    device = "cuda"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=64,  # the rank of the LoRA matrices
        lora_alpha=16, # the weight
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
        modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
    )

    training_args = SFTConfig(
        output_dir= parent_directory + input,
        # report_to="wandb",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio = 0.1,
        gradient_checkpointing=True,
        # eval_strategy="epoch",
        num_train_epochs=epochs,
        # logging strategies 
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        max_seq_length=2048,
        packing=True,
        run_name=input,
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config = peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"{parent_directory}/{input}_comm_{args.comm}")
    trainer.model.save_pretrained(f"{parent_directory}/ckp_comm_{args.comm}")
    tokenizer.save_pretrained(f"{parent_directory}/ckp_comm_{args.comm}")

    merge_peft_adapters(adapter_dir=f"{parent_directory}/{input}_comm_{args.comm}",
                        output_path=f"{parent_directory}/sft_merged_ckp_{args.comm}")
    
