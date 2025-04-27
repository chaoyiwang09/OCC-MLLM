import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import argparse
import yaml
import re 
import logging
from tqdm import tqdm
import json
from typing import List
import os
import sys
from transformers import Trainer
from dataclasses import dataclass, field
#sys.path.append('/ailab-train/llm/mayangyang/SCoRe')

#from Evaluator.utils import extract_answer_from_completion, is_equiv
@dataclass
class BatchInfo:
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

class ScTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # 先保存自定义配置
        self.config = kwargs.pop('config', None)
        
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)
        
        # 从 kwargs 获取其他自定义配置
        self.model = kwargs.get('model')
        self.ref_model = kwargs.get('ref_model')
        self.tokenizer = kwargs.get('tokenizer')
        self.optimizer = kwargs.get('optimizer')
        self.scheduler = kwargs.get('scheduler')
        self.train_loader = kwargs.get('train_loader')
        self.val_loader = kwargs.get('val_loader')
        self.logger = kwargs.get('logger')
        
        # 初始化其余属性
        self.global_step = 0
        self.reward_history: List[float] = []
        self.edit_distance_ratios: List[float] = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
        
        # 从 config 中加载相关参数
        self.beta1: float = self.config.beta_1
        self.beta2: float = self.config.beta_2
        self.alpha: float = self.config.alpha
        self.train_stage_type = self.config.train_stage_type
        self.online_writeout_stage_one_path = self.config.online_writeout_stage_one_path
        
        if os.path.isdir(self.online_writeout_stage_one_path):
            self.online_writeout_stage_one_path = os.path.join(
                self.online_writeout_stage_one_path, 'stage_one_online_data_writeout.jsonl'
            )
            
        self.model_type = self.config.model_type
        
        # 自定义阶段计数
        self.stage_1_step = 0
        self.stage_2_step = 0
        
        # 如果需要动态初始化额外属性
        if self.train_stage_type == "specific_type":
            self.running = RunningMoments(self.config.some_other_attribute)

        self.current_batch_info = BatchInfo()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        batch_size = inputs['input_ids'].shape[0]
        questions = [self.train_dataset[i]['question'] for i in range(batch_size)]
        answers = [self.train_dataset[i]['answer'] for i in range(batch_size)]
    
        self.current_batch_info.questions = questions
        self.current_batch_info.answers = answers
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # First attempt - use no_grad for generate
        with torch.no_grad():
            first_attempt_outputs = model.generate(
                input_ids=inputs['input_ids'].to(device),
                labels=inputs['labels'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                position_ids=inputs['position_ids'].to(device),
                pixel_values=inputs['pixel_values'].to(torch.bfloat16).to(device),
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get responses and compute rewards
        first_attempt_responses = self.tokenizer.batch_decode(first_attempt_outputs.sequences, skip_special_tokens=True)
        first_attempt_rewards = self.reward_function(answers, first_attempt_responses)
        norm_first_attempt_rewards = (first_attempt_rewards - first_attempt_rewards.mean()) / (first_attempt_rewards.std() + 1e-6)
        
        # Second attempt - use no_grad for generate
        with torch.no_grad():
            second_attempt_outputs = model.generate(
                input_ids=inputs['input_ids'].to(device),
                labels=inputs['labels'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                position_ids=inputs['position_ids'].to(device),
                pixel_values=inputs['pixel_values'].to(torch.bfloat16).to(device),
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Compute second rewards
        second_attempt_responses = self.tokenizer.batch_decode(second_attempt_outputs.sequences, skip_special_tokens=True)
        second_attempt_rewards = self.reward_function(answers, second_attempt_responses)
        bonuses = self.alpha * (second_attempt_rewards - first_attempt_rewards)
        second_attempt_rewards = second_attempt_rewards + bonuses
        norm_second_attempt_rewards = (second_attempt_rewards - second_attempt_rewards.mean()) / (second_attempt_rewards.std() + 1e-6)
        
        # Single forward pass for computing logits
        outputs = model(**inputs)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Compute REINFORCE losses
        first_attempt_reinforce_loss = self.compute_reinforce_loss(
            log_probs.clone(), 
            first_attempt_outputs, 
            norm_first_attempt_rewards,
            batch_size
        )
        
        second_attempt_reinforce_loss = self.compute_reinforce_loss(
            log_probs,
            second_attempt_outputs,
            norm_second_attempt_rewards,
            batch_size
        )
        
        # Compute KL loss with detached reference
        kl_loss = self.compute_kl_divergence(logits, logits.detach())
        
        # Compute total loss
        #l2_reg = sum(torch.sum(p ** 2) for p in model.parameters()) * 1e-4
        total_loss = abs(first_attempt_reinforce_loss + second_attempt_reinforce_loss + self.beta1 * kl_loss)
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def compute_reinforce_loss(self, log_probs, response_ids, rewards, batch_size):
        #print("Original shapes:")
        #print("log_probs shape:", log_probs.shape)
        #print("response_ids type:", type(response_ids))
        #print("response_ids.sequences shape:", response_ids.sequences.shape)
        
        losses = []
        
        for i in range(batch_size):
            # Get current batch data
            current_log_probs = log_probs[i]  # [363, 32020]
            current_ids = response_ids.sequences[i].to(log_probs.device).unsqueeze(0)  # [1, 257]
            
            # Adjust sequence length
            seq_length = min(current_log_probs.size(0), current_ids.size(1))
            current_log_probs = current_log_probs[:seq_length]  # [seq_length, 32020]
            current_ids = current_ids[:, :seq_length]  # [1, seq_length]
            
            # Create padding mask
            padding_mask = (current_ids != self.tokenizer.pad_token_id).float()  # [1, seq_length]
            
            # Gather operation
            selected_log_probs = current_log_probs.gather(
                dim=-1, 
                index=current_ids  # [1, seq_length]
            )  # [1, seq_length]
            
            # Apply padding mask
            masked_log_probs = selected_log_probs * padding_mask  # [1, seq_length]
            # Make sure rewards is on the correct device
            current_reward = rewards[i].to(masked_log_probs.device)
            losses.append((-masked_log_probs * current_reward).sum())
        
        # Ensure we're creating a fresh computation graph for the final loss
        final_loss = torch.stack(losses).mean()
        
        return final_loss
    
    def compute_kl_divergence(self, logits, ref_logits):
        """计算KL散度"""
        log_probs = F.log_softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        return F.kl_div(log_probs, ref_probs, reduction="batchmean")
    
    def reward_function(self, gt_answers: List[str], generated_answers: List[str]) -> torch.Tensor:
        """计算奖励值"""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        rewards = []
        for gt, gen in zip(gt_answers, generated_answers):
            reward = 1.0 if self.extract_answer(gt) == self.extract_answer(gen) else \
    0.1 + 0.9 * self.string_similarity(self.extract_answer(gt), self.extract_answer(gen))
            rewards.append(reward)
        return torch.tensor(rewards,device = device)
    
    @staticmethod
    def extract_answer(text: str) -> str:
        """从文本中提取答案"""
        return text.lower().strip()
        
    @staticmethod
    def string_similarity(str1, str2):
        from Levenshtein import distance
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0  # Both strings are empty
        return 1 - (distance(str1, str2) / max_len)