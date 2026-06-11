import json
import random
from pathlib import Path
import os
import torch
import gc
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from torch.utils.data import DataLoader    
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from tests.adapters import run_grpo_train_step as grpo_train_step
from cs336_alignment.vllm_utils import VLLMServer 
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn 


MODEL_DIR = "/home/ec2-user/assignment5-alignment/cs336_alignment/models"
MODEL_ID= "allenai/OLMo-2-0425-1B"
DATA_DIR = "/home/ec2-user/assignment5-alignment/data/gsm8k"
PROMPT_DIR = "/home/ec2-user/assignment5-alignment/cs336_alignment/prompts"


n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 32*2
group_size = 2
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0


class MathDataset(Dataset):
    def __init__(self, dataset_path ):
        self.data = []
        with open( dataset_path) as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)      

class GenerateRollout:
    def __init__(self, llm, group_size,prompt_template_path):
        self.llm = llm 
        self.group_size = group_size
        self.batch_size=8
        self.llm = llm 
        self.group_size=group_size
        self.sampling_params = {
            "temperature": 1.0,
            "stop":  ["</answer>"],
            "include_stop_str_in_output": True,
            "top_p": 1,
            "max_tokens": 512,
            "n":self.group_size,
            "seed": 42
        }
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()
    
    def __call__(self, samples):
        self.llm.start()
        repeated_prompts= [self.prompt_template.format(**line) for line in samples]
        output = self.llm.generate_completions(
            prompts =repeated_prompts,
            sampling_params=self.sampling_params,
            batch_size=self.batch_size,
        )
        self.llm.stop()
        return {
            "rollout_responses": [x.text for x in output],
            "repeated_prompts": [x for x in repeated_prompts for _ in range(self.group_size)],
            "repeated_ground_truths": [x['answer'].split("####")[1].strip() for x in samples for _ in range(self.group_size)]
        }




if __name__ == "__main__":

    #initialize model .... 
    model_path = f"{MODEL_DIR}/{MODEL_ID}"
    if not Path(model_path).exists():
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_path,
        )

    llm = VLLMServer(model_id=model_path,gpu=0)
    train_dataset = MathDataset(f"{DATA_DIR}/train.jsonl")
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size//group_size,shuffle=True,collate_fn=GenerateRollout(llm,group_size,f"{PROMPT_DIR}/r1_zero_three_shot_gsm8k.prompt"))
    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    train_step = 0
    policy = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
            )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    policy = get_peft_model(policy, lora_config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)      
    
    while train_step < num_rollout_steps:
        for batch in train_dataloader:
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            policy.to(device_gpu)
            optimizer_to(optimizer, device_gpu)
            grpo_train_step(
                model= policy, optimizer=optimizer, tokenizer=tokenizer, gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm, reward_fn=r1_zero_reward_fn,
                rollout_responses=batch["rollout_responses"],repeated_prompts=batch["repeated_prompts"],repeated_ground_truths=batch["repeated_ground_truths"],group_size=group_size,baseline='mean',advantage_normalizer="std",
            )        
            #save_weights
            policy.save_pretrained(model_path)
            policy.to(device_cpu)
            optimizer_to(optimizer, device_cpu)
            gc.collect()
            torch.cuda.empty_cache()
            train_step+=1

        




