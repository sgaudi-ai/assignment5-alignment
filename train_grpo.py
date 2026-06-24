import json
import random
from pathlib import Path
import os
import torch
import gc
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset
from torch.utils.data import DataLoader    
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tests.adapters import run_grpo_train_step as grpo_train_step, run_compute_rollout_rewards
from cs336_alignment.vllm_utils import VLLMServer 
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn 
import bitsandbytes as bnb 
import random




MODEL_DIR = "/home/ec2-user/assignment5-alignment/models"
CHECKPOINT_DIR = "/home/ec2-user/assignment5-alignment/checkpoints"
MODEL_ID= "allenai/OLMo-2-0425-1B"
DATA_DIR = "/home/ec2-user/assignment5-alignment/data/gsm8k"
PROMPT_DIR = "/home/ec2-user/assignment5-alignment/cs336_alignment/prompts"


n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 800
learning_rate =  1e-5
group_size = 4
gradient_accumulation_steps = 8
val_batch_size =  32
rollout_batch_size = train_batch_size = group_size*gradient_accumulation_steps
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
        
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

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
            "max_tokens": 1024,
            "n":self.group_size,
            "seed": random.randint(0, 2**31 - 1),
        }
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()
    
    def __call__(self, samples):
        self.llm.start()

        repeated_prompts= [self.prompt_template.format(question=line) for line in samples['question']]
        output = self.llm.generate_completions(
            prompts =repeated_prompts,
            sampling_params=self.sampling_params,
            batch_size=self.batch_size,
        )
        self.llm.stop()
        gc.collect()
        del self.llm
        torch.cuda.empty_cache()
        return {
            "rollout_responses": [x.text for x in output],
            "repeated_prompts": [x for x in repeated_prompts for _ in range(self.group_size)],
            "repeated_ground_truths": [x.split("####")[1].strip() for x in samples['answer'] for _ in range(self.group_size)]
        }


def evaluate(val_dataloader,prompt_template_path,llm,reward_fn,num_of_samples):
        sampling_params = {
            "temperature": 1.0,
            "stop":  ["</answer>"],
            "include_stop_str_in_output": True,
            "top_p": 0.95,
            "max_tokens": 1024,
            "n":1,
            "seed": 42,
        }
        with open(prompt_template_path) as f:
            prompt_template = f.read()
        
        llm.start()
        total_samples,rewards,answer_reward,format_reward  = 0,0,0,0

        for samples in val_dataloader:
            if total_samples > num_of_samples:
                break
            prompts= [prompt_template.format(question=line) for line in samples['question']]
            output = llm.generate_completions(
                prompts =prompts,
                sampling_params=sampling_params,
                batch_size= len(prompts),
            )
            raw_rewards,rewards_metadata = run_compute_rollout_rewards(reward_fn,  [x.text for x in output] ,[x.split("####")[1].strip() for x in samples['answer']])
            total_samples +=  len(prompts)
            rewards += raw_rewards.sum()
            answer_reward += rewards_metadata['mean_answer']*rewards_metadata['count']
            format_reward += rewards_metadata['mean_format']*rewards_metadata['count']
        
        llm.stop()
        gc.collect()
        del llm
        torch.cuda.empty_cache()
        return {
            "total_reward" : rewards / total_samples,
            "answer_reward": answer_reward / total_samples,
            "format_reward": format_reward / total_samples
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

    train_dataset = MathDataset(f"{DATA_DIR}/train.jsonl")
    val_dataset = MathDataset(f"{DATA_DIR}/test.jsonl")
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size//group_size,shuffle=True,num_workers=0,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=val_batch_size,shuffle=False,num_workers=0)
    
    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    train_step = 0
    policy = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa"
            )
    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)      
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    while train_step < num_rollout_steps:
        for batch in train_dataloader:
            
            llm = VLLMServer(model_id=model_path, gpu=0)
            if (train_step +1) %100==0:
                evaluation_metadata = evaluate(val_dataloader,f"{PROMPT_DIR}/r1_zero.prompt", llm, r1_zero_reward_fn,num_of_samples=500)
                print("="*50)
                print(evaluation_metadata)
                print("="*50)
            rollout_fn = GenerateRollout(llm,group_size,f"{PROMPT_DIR}/r1_zero.prompt")
            batch = rollout_fn(batch)
            print(batch["repeated_prompts"][0],batch["rollout_responses"][0])
            print(batch["repeated_ground_truths"][0])
            policy.to(device_gpu)
            policy.train()
            optimizer_to(optimizer, device_gpu) 
            optimizer.zero_grad(set_to_none=True)           
            loss, all_metadata = grpo_train_step(
                model= policy, optimizer=optimizer, tokenizer=tokenizer, gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm, reward_fn=r1_zero_reward_fn,
                rollout_responses=batch["rollout_responses"],repeated_prompts=batch["repeated_prompts"],repeated_ground_truths=batch["repeated_ground_truths"],group_size=group_size,baseline='mean',advantage_normalizer="std",
            )
            # print(all_metadata)   
            del  loss, all_metadata    
            #save_weights
            policy.to(device_cpu)
            optimizer_to(optimizer, device_cpu)            
            model_path = f"{CHECKPOINT_DIR}/model"
            os.makedirs(model_path, exist_ok=True)
            policy.save_pretrained(model_path,safe_serialization=True)
            tokenizer.save_pretrained(model_path)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            train_step+=1


        




