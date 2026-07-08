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
from cs336_alignment.drgrpo_grader import grade 
from functools import partial

import bitsandbytes as bnb 
import random
from torch.utils.tensorboard import SummaryWriter


MODEL_DIR = "/home/ec2-user/assignment5-alignment/models"
RUN_DIR = "/home/ec2-user/assignment5-alignment/runs"
MODEL_ID= "allenai/OLMo-2-0425-1B"
DATA_DIR = "/home/ec2-user/assignment5-alignment/data/gsm8k"
PROMPT_DIR = "/home/ec2-user/assignment5-alignment/cs336_alignment/prompts"


n_train_examples = 6400
n_val_examples = 256
num_rollout_steps = 800
learning_rate =  1e-5
group_size = 8
gradient_accumulation_steps = 1
val_batch_size =  32
train_batch_size = rollout_batch_size = 256
sampling_temperature = 1.0
sampling_max_tokens = 800
max_grad_norm = 1.0
offpolicy_clip = 0.2
offpolicy_gspo = 3e-4


class MathDataset(Dataset):
    def __init__(self, dataset_path, tokenizer ):
        self.data = []
        self.tokenizer = tokenizer
        with open( dataset_path) as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return {
            "question": self.tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant internally reasons about the problem but does not reveal its reasoning. It provides only the final answer. If a brief explanation is appropriate, it should be concise and should not include the internal reasoning process. The final answer should be enclosed within \\boxed{}, and no explanation should be enclosed within \\boxed{}.",
            },
            {
                "role": "user", 
                "content":  self.data[idx]["question"]
            }
        ], tokenize=False,add_generation_prompt=True),
        "answer": self.data[idx]["answer"]
        } 
        
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
    def __init__(self, llm, group_size):
        self.llm = llm 
        self.group_size = group_size
        self.batch_size=8
        self.llm = llm 
        self.group_size=group_size
        self.sampling_params = {
            "temperature": 1.0,
            "stop":  ["<|im_end|>"],
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "top_p": 0.95,
            "max_new_tokens": sampling_max_tokens,
            "n":self.group_size,
            "seed": random.randint(0, 2**31 - 1),
            "logprobs": True
        }
    
    def __call__(self, samples):
        self.llm.start()

        repeated_prompts=  samples['question']
        output = self.llm.generate_completions(
            prompts =repeated_prompts,
            sampling_params=self.sampling_params,
            batch_size=self.batch_size,
        )
        self.llm.stop()
        gc.collect()
        del self.llm
        torch.cuda.empty_cache()
        out_repeated_prompts = [x for x in repeated_prompts for _ in range(self.group_size)]
        
        return {
            "rollout_token_ids": [x.token_ids for x in output],
            "repeated_prompts": out_repeated_prompts,
            "repeated_ground_truths": [x.split("####")[1].strip() for x in samples['answer'] for _ in range(self.group_size)],
            "logprobs": [x.logprobs for x in output]
        }

def logprobs_to_padded_tensor(output, pad_value=0.0):
    """
    Converts a list of variable-length logprob lists into a padded tensor.

    Args:
        output: iterable where each element has a .logprobs attribute
        pad_value: value used for padding

    Returns:
        tensor: (batch_size, max_length)
        lengths: original lengths
    """
    logprobs = [torch.as_tensor(x, dtype=torch.float32) for x in output]
    padded = torch.nn.utils.rnn.pad_sequence(
        logprobs,
        batch_first=True,
        padding_value=pad_value,
    )
    return padded

def evaluate(val_dataloader,llm,reward_fn,num_of_samples):
        sampling_params = {
            "temperature": 1.0,
            "stop":  ["<|im_end|>"],
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "top_p": 0.95,
            "max_new_tokens": sampling_max_tokens,
            "n":1,
            "seed": 42,
            "logprobs": False
        }
        
        llm.start()
        total_samples,rewards,num_of_tokens  = 0,0,0
        examples= []

        for samples in val_dataloader:

            if total_samples >= num_of_samples:
                break
            output = llm.generate_completions(
                prompts =  samples['question'],
                sampling_params=sampling_params,
                batch_size= len(samples['question']),
            )
            examples.append({
                "prompt": str(samples['question'][0]),
                "response": output[0].text,
                "gt": samples['answer'][0]
            })
            reward_dict = [reward_fn(resp,gt) for resp,gt in zip([x.text for x in output] ,[x.split("####")[1].strip() for x in samples['answer']] )]
            raw_rewards = torch.tensor([x for x in reward_dict])
            num_of_tokens += sum([len(x.token_ids) for x in output])
            total_samples += len(samples['question'])
            rewards += raw_rewards.sum()
        
        llm.stop()
        gc.collect()
        del llm
        torch.cuda.empty_cache()
        return {
            "total_reward" : rewards / total_samples,
            "avg_token_count": num_of_tokens / total_samples,
            "examples": examples
            }


def grade_reward_func(response, ground_truth, fast=False):
    reward = float(grade(response,ground_truth,fast))
    return {
                "format_reward": 0.0,
                "answer_reward": reward,
                "reward": reward
            }

if __name__ == "__main__":

    
    exp_name = f"test_grpo_after_sft"

    #initialize model .... 

    model_path = f"{RUN_DIR}/test_sft/checkpoints"
    if not Path(model_path).exists():
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_path,
        )


    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    writer = SummaryWriter(log_dir=f"{RUN_DIR}/{exp_name}/tensorboard/")

    train_step,outer_step = 0,0
    policy = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa"
            )
    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)      
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


    train_dataset = MathDataset(f"{DATA_DIR}/train.jsonl",tokenizer=tokenizer)
    val_dataset = MathDataset(f"{DATA_DIR}/test.jsonl",tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size//group_size,shuffle=True,num_workers=0,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size=val_batch_size,shuffle=False,num_workers=0)


    while train_step < num_rollout_steps:
        for batch in train_dataloader:
            
            llm = VLLMServer(model_id=model_path, gpu=0)
            if (outer_step+1) %10 ==0:
                evaluation_metadata = evaluate(val_dataloader, llm, partial(grade, fast=False),num_of_samples=n_val_examples)
                for val_metrics in ["total_reward", "avg_token_count" ]:
                    writer.add_scalar(f"val/{val_metrics}",evaluation_metadata[val_metrics], train_step)
                for i,example in enumerate(evaluation_metadata["examples"]):
                    writer.add_text(tag=f"prompt_{i}",text_string=example['prompt'],global_step=train_step)
                    writer.add_text(tag=f"response_{i}",text_string=example['response'],global_step=train_step)
                    writer.add_text(tag=f"gt_{i}",text_string=example['gt'],global_step=train_step)

            rollout_fn = GenerateRollout(llm,group_size)
            batch = rollout_fn(batch)
            policy.to(device_gpu)

            policy.train()
            optimizer_to(optimizer, device_gpu) 
            optimizer.zero_grad(set_to_none=True)
            for offline_batch_size in range(0, train_batch_size,group_size*gradient_accumulation_steps):         
                rollout_token_ids = batch["rollout_token_ids"][offline_batch_size: offline_batch_size+group_size*gradient_accumulation_steps ]
                repeated_prompts = batch["repeated_prompts"][offline_batch_size: offline_batch_size+group_size*gradient_accumulation_steps ]  
                repeated_ground_truths = batch["repeated_ground_truths"][offline_batch_size: offline_batch_size+group_size*gradient_accumulation_steps ]  
                old_log_probs = logprobs_to_padded_tensor(batch["logprobs"][offline_batch_size: offline_batch_size+group_size*gradient_accumulation_steps ]  )
                loss, all_metadata = grpo_train_step(
                    model= policy,  tokenizer=tokenizer,optimizer=optimizer, gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm, reward_fn=grade_reward_func,
                    rollout_token_ids=rollout_token_ids,repeated_prompts=repeated_prompts,repeated_ground_truths=repeated_ground_truths,group_size=group_size,
                    baseline='mean',advantage_normalizer="mean",loss_normalization="constant",normalization_constant=sampling_max_tokens*group_size,
                    importance_reweighting_method="grpo", cliprange=offpolicy_clip,old_log_probs=old_log_probs
                ) 
                for train_metrics in ["batch_loss","rewards","answer_reward","format_reward","token_entropy","grad_norm", "avg_token_count" ]:
                    writer.add_scalar(f"train/{train_metrics}",all_metadata[train_metrics], train_step)
                
                del  loss, all_metadata    
                train_step+=1
            outer_step+=1

            #save_weights
            policy.to(device_cpu)
            optimizer_to(optimizer, device_cpu)            
            model_path = f"{RUN_DIR}/{exp_name}/checkpoints"
            os.makedirs(model_path, exist_ok=True)
            policy.save_pretrained(model_path,safe_serialization=True)
            tokenizer.save_pretrained(model_path)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            


        




