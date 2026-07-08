

import os
from string import Template
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers import get_linear_schedule_with_warmup
from cs336_alignment.drgrpo_grader import grade 
from cs336_alignment.vllm_utils import VLLMServer 

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import torch
from torch import Tensor
import json

import bitsandbytes as bnb 
from torch.utils.tensorboard import SummaryWriter
import gc

from train_grpo import optimizer_to
from functools import partial



train_batch_size = 2
val_batch_size = 32
learning_rate = 1e-5
gradient_accumulation_steps = 8
max_grad_norm = 1.0
train_steps = 75_000
warmup_steps = int(0.03 * train_steps)  
val_examples = 128
sampling_max_tokens = 512
max_length = 4096

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



MODEL_DIR = "/home/ec2-user/assignment5-alignment/models"
RUN_DIR = "/home/ec2-user/assignment5-alignment/runs"
MODEL_ID= "allenai/OLMo-2-0425-1B"
DATA_DIR = "/home/ec2-user/assignment5-alignment/data"
PROMPT_DIR = "/home/ec2-user/assignment5-alignment/cs336_alignment/prompts"

class TrainSFTDataset(Dataset):
    def __init__(self,datasetpath, tokenizer):
        self.dataset = load_dataset(datasetpath)['train']
        self.tokenizer=  tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        messages = [
            {
                "role": "system",
                "content": "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant internally reasons about the problem but does not reveal its reasoning. It provides only the final answer. If a brief explanation is appropriate, it should be concise and should not include the internal reasoning process. The final answer should be enclosed within \\boxed{}, and no explanation should be enclosed within \\boxed{}.",
            },
            {
                "role": "user", 
                "content": item["instruction"]
            },
            {
                "role": "assistant",
                "content": item["output"]
            }
        ]
        return {
            "prompt_str" : self.tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=False),
            "response_str" : self.tokenizer.apply_chat_template(messages[-1:],tokenize=False,add_generation_prompt=False)
        }
    
        

            

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
        "answer":     self.data[idx]["answer"]
        } 

def run_tokenize_prompt_and_output(
    prompt_str: list[str],
    responses_str: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, Tensor]:
    input_id_len = tokenizer(prompt_str,padding=False,add_special_tokens=False,return_length=True,truncation=True, max_length=max_length)["length"]
    output_ids = tokenizer([p+r for p,r in zip(prompt_str,responses_str)],padding=True,add_special_tokens=False,truncation=True, max_length=max_length)
    return {
        "input_ids": torch.tensor(output_ids.input_ids)[:, :-1],
        "labels": torch.tensor(output_ids.input_ids)[:, 1:],
        "attention_mask": torch.tensor(output_ids.attention_mask)[:, :-1],
        "response_mask": torch.tensor([
            [0] * (l - 1) + m[l:]
            for l, m in zip(input_id_len, output_ids.attention_mask)
        ]),
    }
        


if __name__ == "__main__":
    """    
    1. Distilling a powerful model. Filter the data and check if it does not contain val dataset (LazyAGI/GSM8K_Deepseek_R1_Distill-Data-7148)
    2. Verify the traces only contain correct answer.
    3. Finetune on the traces with the prompt. 
    # 
    """
    exp_name = f"test_sft"

    model_path = f"{MODEL_DIR}/{MODEL_ID}"
    if not Path(model_path).exists():
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=model_path,
        )


    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    writer = SummaryWriter(log_dir=f"{RUN_DIR}/{exp_name}/tensorboard/")
    policy = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            ).to(device_gpu)
    policy.train()
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    optimizer = bnb.optim.AdamW8bit(policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)   

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=train_steps,
    # )
    optimizer_to(optimizer, device_gpu)    
    optimizer.zero_grad(set_to_none=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    train_dataset = TrainSFTDataset(f"{DATA_DIR}/GSM8K_Deepseek_R1_Distill-Data-7148", tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True,num_workers=0,drop_last=True)
    val_dataset = MathDataset(f"{DATA_DIR}/gsm8k/test.jsonl", tokenizer=tokenizer)
    val_dataloader = DataLoader(val_dataset,batch_size=val_batch_size,shuffle=False,num_workers=0)
    curr_train_step = 0
    while curr_train_step < train_steps:
        for batch in train_dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokenized_input = run_tokenize_prompt_and_output(batch["prompt_str"],batch["response_str"],tokenizer,max_length=max_length)
                input_ids, labels,response_mask,attention_mask  = tokenized_input["input_ids"].to(device_gpu), tokenized_input["labels"].to(device_gpu), tokenized_input["response_mask"].to(device_gpu),tokenized_input["attention_mask"].to(device_gpu)
                logits= policy(input_ids,attention_mask = attention_mask).logits
                B,C,V = logits.shape
                token_loss = F.cross_entropy(
                        logits.view(-1, V),
                            labels.view(-1),
                            reduction="none",
                        ).view(B, C)
                example_loss = (
                    token_loss * response_mask
                ).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)

                loss = example_loss.mean()
                writer.add_scalar(f"train/loss",loss.item(), curr_train_step)

            loss.backward()
            del loss
            if (curr_train_step +1) % gradient_accumulation_steps == 0:
                
                grad_norm = clip_grad_norm_(
                        policy.parameters(),
                        max_norm=max_grad_norm
                    )
                writer.add_scalar(f"train/grad_norm",grad_norm.item(), curr_train_step)
                
                
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
            if (curr_train_step+1) %1000 ==0:
                
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

                llm = VLLMServer(model_id=model_path, gpu=0)
                evaluation_metadata = evaluate(val_dataloader, llm, partial(grade, fast=False),num_of_samples=val_examples)
                for val_metrics in ["total_reward", "avg_token_count" ]:
                    writer.add_scalar(f"val/{val_metrics}",evaluation_metadata[val_metrics], curr_train_step)
                for i,example in enumerate(evaluation_metadata["examples"]):
                    writer.add_text(tag=f"prompt_{i}",text_string=example['prompt'],global_step=curr_train_step)
                    writer.add_text(tag=f"response_{i}",text_string=example['response'],global_step=curr_train_step)
                    writer.add_text(tag=f"gt_{i}",text_string=example['gt'],global_step=curr_train_step)

                policy.to(device_gpu)

                policy.train()
                policy.gradient_checkpointing_enable()
                policy.config.use_cache = False
                optimizer_to(optimizer, device_gpu) 
                optimizer.zero_grad(set_to_none=True)


            curr_train_step +=1



        
        
        

