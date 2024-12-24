import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
# Set environment variables for distributed training
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["WORLD_SIZE"] = "5"  # Number of GPUs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# os.environ["RANK"] = "0"

# 用于生成数据的函数
def generate_data_on_gpu(model, tokenizer, device, prompt, max_length=50):
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# 用于分布式训练或生成的初始化函数
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理函数
def cleanup():
    dist.destroy_process_group()

# 主训练/生成函数
def main(rank, world_size):
    setup(rank, world_size)

    # 初始化模型和分布式数据并行
    model_name = "/home/fanqi/llm_simulation/models/sft_model/final_merged_checkpoint"  # 替换为你的模型路径
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device(f"cuda:{rank}")
    model.to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 生成数据并将其存储在列表中
    generated_data = []
    prompt = "hello! I am a student, how are you today?"

    for _ in range(4):  # 循环128次
        # 在每个 GPU 上生成1条数据
        generated_text = generate_data_on_gpu(model, tokenizer, rank, prompt)
        generated_data.append(generated_text)

    print(f"Rank {rank} generated {len(generated_data)} texts.")
    # 汇总所有生成的数据
    all_generated_data = gather_results(generated_data, rank, world_size)

    # 在主进程打印所有生成的数据
    if rank == 0:
        print(f"All generated texts: {all_generated_data}")
    
    cleanup()

# 将不同 GPU 上生成的数据汇总到主进程（rank=0）
def gather_results(local_data, rank, world_size):
    gathered_data = None
    if rank == 0:
        gathered_data = [None] * world_size
    dist.gather_object(local_data, gathered_data, dst=0)
    return gathered_data

if __name__ == "__main__":
    world_size = 5  # 8 张 GPU
    # 使用 torch.multiprocessing 启动多个进程，每个进程使用一个 GPU
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
