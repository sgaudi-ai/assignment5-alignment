from __future__ import annotations

import os
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
import math


import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    rollout_token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    
    prompt_tokens = tokenizer(prompt_strs,padding=False,add_special_tokens=False)
    output_tokens = rollout_token_ids
    input_ids = [p+r for p,r in zip(prompt_tokens['input_ids'],output_tokens)]
    max_pad = [tokenizer.pad_token_id]* max([len(i) for i in input_ids])
    response_mask = [[0]*len(p) + [1]*len(r) + [0]*len(max_pad[len(p)+len(r): ]) for p,r in zip(prompt_tokens['input_ids'],output_tokens)]
    input_ids = [i + max_pad[len(i): ]  for i in input_ids ]
    labels =     [i[1:]  for i in input_ids ]
    response_mask =  [r[1:]  for r in response_mask ]
    input_ids =  [i[:-1]  for i in input_ids ]
    return {
        "input_ids": torch.tensor(input_ids),
        "response_mask": torch.tensor(response_mask),
        "labels": torch.tensor(labels)
    }
    
 


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    logits = model(input_ids).logits
    log_p_theta_x_t_giv_x_less_t =  torch.log_softmax(logits, dim=-1)
    p_theta_x_t_giv_x_less_t = torch.exp(log_p_theta_x_t_giv_x_less_t)
    B,S,_ = log_p_theta_x_t_giv_x_less_t.shape
    return_dict = {

        "log_probs": log_p_theta_x_t_giv_x_less_t.reshape(B*S,-1)[torch.arange(0,B*S),labels.reshape(B*S)].reshape(B,S),
    }   
    if return_token_entropy:
        return_dict["token_entropy"] = -1*(p_theta_x_t_giv_x_less_t * log_p_theta_x_t_giv_x_less_t).sum(dim=-1)
    return return_dict

    

    

    


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    reward_dict = [reward_fn(resp,gt) for resp,gt in zip(rollout_responses,repeated_ground_truths )]
    raw_rewards = torch.tensor([x["answer_reward"]*1 for x in reward_dict])
    metadata = {
        "mean_total": raw_rewards.mean().item(),
        "mean_format":torch.tensor([x["format_reward"] for x in reward_dict]).mean().item(),
        "mean_answer": torch.tensor([x["answer_reward"] for x in reward_dict]).mean().item(),
        "raw_rewards": raw_rewards,
        "count": len(reward_dict)
    }
    return raw_rewards,metadata 



def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    raw_rewards = raw_rewards.reshape(-1,group_size) 
    advantages = raw_rewards
    mean_rewards = None
    if baseline == "mean" or advantage_normalizer=="mean":
        mean_rewards = raw_rewards.mean(dim=-1, keepdim=True)
    if baseline == "mean":
        advantages = advantages - mean_rewards
    if advantage_normalizer == "std":
        std_rewards = raw_rewards.std(unbiased=True, dim=-1, keepdim=True)
        advantages = advantages / std_rewards.clamp_min(advantage_eps) 
    if advantage_normalizer == "mean":
        advantages = advantages / mean_rewards.clamp_min(advantage_eps)
    return advantages.reshape(-1), {}




        



def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    if  importance_reweighting_method == "none":
        return -1*raw_rewards_or_advantages.reshape(-1,1)*policy_log_probs ,{}
    if importance_reweighting_method in {"noclip"}:
        w = raw_rewards_or_advantages.reshape(-1,1)*torch.exp( policy_log_probs - old_log_probs)
        return -1*w, {}
    elif importance_reweighting_method =="grpo":
        w = torch.exp( policy_log_probs - old_log_probs)
        w =  torch.min(raw_rewards_or_advantages.reshape(-1,1)*w,raw_rewards_or_advantages.reshape(-1,1)*torch.clip(w, 1- cliprange, 1+cliprange))
        return -1*w, {}
    if importance_reweighting_method in { "gspo"}:
        w = torch.exp( policy_log_probs - old_log_probs)
        w = torch.pow(torch.prod(torch.pow(w, response_mask), dim=-1), 1/response_mask.sum(dim=-1)).reshape(-1,1)
        w =  torch.min(raw_rewards_or_advantages.reshape(-1,1)*w,raw_rewards_or_advantages.reshape(-1,1)*torch.clip(w, 1- cliprange, 1+cliprange))
        return -1*w*torch.ones_like(policy_log_probs), {}

  
    



def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    loss= (per_token_policy_gradient_loss *mask)
    if loss_normalization == "sequence":
        loss = (loss.sum(dim=-1) / mask.sum(dim=-1)).mean()
    if  loss_normalization == "constant":
        loss = (loss.sum(dim=-1) / normalization_constant).sum()
    return loss


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_token_ids: list[int],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    tokenized_dict = run_tokenize_prompt_and_output(repeated_prompts, rollout_token_ids,tokenizer)
    input_ids, labels,response_mask = tokenized_dict["input_ids"].to(model.device), tokenized_dict["labels"].to(model.device),tokenized_dict["response_mask"].to(model.device)
    microbatch_size = len(input_ids) // gradient_accumulation_steps
    
    batch_loss = 0
    rewards,format_reward, answer_reward,token_entropy = 0,0,0,0
    advantage_metadata,per_token_metadata ={},{}
    advantage = torch.empty(len(input_ids), device=model.device)
    for i in range(0,len(input_ids),group_size):
        raw_rewards,rewards_metadata = run_compute_rollout_rewards(reward_fn, tokenizer.decode(rollout_token_ids[i:i+group_size]),repeated_ground_truths[i:i+group_size])
        _advantage, advantage_metadata = run_compute_group_normalized_rewards(raw_rewards.to(model.device), group_size,baseline=baseline,advantage_normalizer=advantage_normalizer,advantage_eps=advantage_eps)     
        advantage[i:i+group_size] = _advantage
        
        with torch.no_grad():
            rewards += raw_rewards.sum().item()
            answer_reward += rewards_metadata['mean_answer']*rewards_metadata['count']
            format_reward += rewards_metadata['mean_format']*rewards_metadata['count']

    for i in range(0, len(input_ids), microbatch_size):
        log_probs = run_get_response_log_probs(model,input_ids[i:i+microbatch_size] , labels[i:i+microbatch_size],return_token_entropy= True)
        if old_log_probs is None: 
            curr_old_log_probs = None
        else:
            curr_old_log_probs = old_log_probs[i:i+microbatch_size].to(model.device)
        per_token_policy_gradient_loss, per_token_metadata = run_compute_policy_gradient_loss(advantage[i:i+microbatch_size],log_probs["log_probs"],response_mask=response_mask[i:i+microbatch_size],importance_reweighting_method= importance_reweighting_method, old_log_probs=curr_old_log_probs , cliprange=cliprange)
        loss = run_aggregate_loss_across_microbatch(per_token_policy_gradient_loss,response_mask[i:i+microbatch_size],loss_normalization=loss_normalization,normalization_constant=normalization_constant)
        
        if loss_normalization in {"sequence"}:
            loss = loss*microbatch_size / len(input_ids)

        
        loss.backward()
        with torch.no_grad():
            batch_loss +=loss
            token_entropy += (((log_probs["token_entropy"]*response_mask[i:i+microbatch_size]).sum(dim=-1) / response_mask[i:i+microbatch_size].sum(dim=-1)) / math.log(len(tokenizer.vocab))).sum()
    
    grad_norm = clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm
    )
    # Update weights once across entire batch.
    optimizer.step()
    # Zero gradients once across entire batch.
    optimizer.zero_grad()

    with torch.no_grad():
        all_metadata = {
            "rewards_metadata": rewards_metadata,
            "advantage_metadata": advantage_metadata,
            "per_token_metadata":per_token_metadata,
            "batch_loss":batch_loss.item(),
            "rewards": rewards / len(input_ids),
            "answer_reward": answer_reward / len(input_ids),
            "format_reward": format_reward  / len(input_ids),
            "token_entropy": token_entropy / len(input_ids),
            "grad_norm": grad_norm.item(),
            "avg_token_count": response_mask.sum(dim=-1).float().mean().item()
        }
    return batch_loss, all_metadata

def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
