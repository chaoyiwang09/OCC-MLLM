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
from transformers import Trainer,AutoModel
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
        self.ref_model = AutoModel.from_pretrained(
            "/root/autodl-tmp/workspace/install/InternVL2-4B/",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        )
        self.tokenizer = kwargs.get('tokenizer')
        self.optimizer = kwargs.get('optimizer')
        self.scheduler = kwargs.get('scheduler')
        self.train_loader = kwargs.get('train_loader')
        self.val_loader = kwargs.get('val_loader')
        self.logger = kwargs.get('logger')
        #self.train_dataset = kwargs.get('train_dataset')
        
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
        self.stage_1_epoch = 1
        self.stage_2_epoch = 1
        
        # 如果需要动态初始化额外属性
        if self.train_stage_type == "specific_type":
            self.running = RunningMoments(self.config.some_other_attribute)

        self.current_batch_info = BatchInfo()
        self.current_iteration = 0

        self.learning_rate = 4e-5
        self.output_dir = "./outputSCcy001"

    def train_step_stage_one(self, batch):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # 获取问题和答案
        questions = [batch.get("question", "")]
        answers = [batch.get("answer", "")]
        batch_size = 1  # 单样本
        
        # 确保ref_model处于评估模式
        self.ref_model.eval()
        
        # 第一次生成尝试（first attempt）- 不需要梯度
        with torch.no_grad():
            first_attempt_outputs = self.model.generate(
                input_ids=batch['input_ids'].unsqueeze(0).to(device),
                attention_mask=batch['attention_mask'].unsqueeze(0).to(device),
                position_ids=batch['position_ids'].unsqueeze(0).to(device),
                pixel_values=batch['pixel_values'].to(torch.bfloat16).to(device),
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # 获取第一次尝试的响应
        first_attempt_responses = self.tokenizer.batch_decode(first_attempt_outputs.sequences, skip_special_tokens=True)
        
        # 计算第一次尝试的奖励
        first_attempt_rewards = self.reward_function(answers, first_attempt_responses)
        
        # 归一化奖励
        norm_first_attempt_rewards = (first_attempt_rewards - first_attempt_rewards.mean()) / (first_attempt_rewards.std() + 1e-6)
        
        # 准备第二次尝试的输入
        second_attempt_prompts = self.tokenizer.decode(batch['input_ids'], skip_special_tokens=False).replace("What's the object in the hand","Please check again what's the object in the hand")
        second_attempt_inputs = self.tokenizer(second_attempt_prompts, return_tensors="pt", truncation=True)
        
        # 将second_attempt_inputs移动到device
        second_attempt_inputs = {k: v.to(device) for k, v in second_attempt_inputs.items()}
        
        # 处理attention mask和position ids
        second_attempt_attention_mask = second_attempt_inputs['attention_mask']
        second_attempt_position_ids = second_attempt_attention_mask.long().cumsum(-1) - 1
        second_attempt_position_ids.masked_fill_(second_attempt_attention_mask == 0, 1)
        
        # 创建完整的batch字典，包含所有必要的键值
        full_batch = {
            'input_ids': second_attempt_inputs['input_ids'],
            'attention_mask': second_attempt_attention_mask,
            'position_ids': second_attempt_position_ids,
            'pixel_values': batch['pixel_values'].to(torch.bfloat16).to(device),
            'image_flags': torch.tensor([1] * batch['pixel_values'].size(0), dtype=torch.long)
        }
        
        # 使用model(**batch)获取outputs - 这会创建计算图
        outputs = self.model(**full_batch)
        
        # 获取参考模型的输出 - 不需要梯度
        with torch.no_grad():
            ref_outputs = self.model(**full_batch)
        
        # 计算KL散度 - 使用outputs.logits而不是scores
        kl_loss = self.compute_kl_divergence(outputs.logits, ref_outputs.logits)
        
        # 使用outputs.logits计算策略梯度损失
        # 这里需要根据outputs.logits计算RL loss
        # 可以使用CrossEntropyLoss来计算策略梯度损失
        # 假设我们有一个target_ids表示目标输出
        
        # 获取第二次尝试的生成结果 - 需要创建计算图
        second_attempt_outputs = self.model.generate(
            input_ids=second_attempt_inputs['input_ids'],
            attention_mask=second_attempt_attention_mask,
            position_ids=second_attempt_position_ids,
            pixel_values=batch['pixel_values'].to(torch.bfloat16).to(device),
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        second_attempt_responses = self.tokenizer.batch_decode(second_attempt_outputs.sequences, skip_special_tokens=True)
        
        # 计算第二次尝试的奖励
        second_attempt_rewards = self.reward_function(answers, second_attempt_responses)
        
        # 计算奖励差异作为bonus
        bonuses = self.alpha * (second_attempt_rewards - first_attempt_rewards)
        
        # 最终第二次尝试的奖励
        second_attempt_rewards = second_attempt_rewards + bonuses
        
        # 归一化第二次尝试的奖励
        #norm_second_attempt_rewards = (second_attempt_rewards - second_attempt_rewards.mean()) / (second_attempt_rewards.std() + 1e-6)
        # 只有一个样本的时候用这一行
        norm_second_attempt_rewards = second_attempt_rewards
        # 计算RL损失 - 使用logits而不是scores
        rl_loss = -torch.mean(outputs.logits.log_softmax(-1) * norm_second_attempt_rewards.unsqueeze(-1).unsqueeze(-1))
        
        # 组合损失
        beta_2 = self.beta2
        total_loss = rl_loss + beta_2 * kl_loss
        #print(rl_loss,beta_2,kl_loss)
        return total_loss, {
            'rl_loss': rl_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'first_rewards': first_attempt_rewards.mean().item(),
            'second_rewards': second_attempt_rewards.mean().item()
        }
    
    def train_step_stage_two(self, batch):
        """
        Stage II 训练步骤实现: 联合优化两次尝试性能
        适用于单样本batch的情况
        
        参数:
        - batch: 包含训练数据的批次（单个样本）
        
        返回:
        - 损失值
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # 获取问题和答案
        questions = [batch.get("question", "")]
        answers = [batch.get("answer", "")]
        batch_size = 1  # 单样本
        
        # 确保ref_model处于评估模式
        self.ref_model.eval()
        
        # 创建第一次尝试的batch
        first_batch = {
            'input_ids': batch['input_ids'].unsqueeze(0).to(device),
            'attention_mask': batch['attention_mask'].unsqueeze(0).to(device),
            'position_ids': batch['position_ids'].unsqueeze(0).to(device),
            'pixel_values': batch['pixel_values'].to(torch.bfloat16).to(device),
            'image_flags': torch.tensor([1] * batch['pixel_values'].size(0), dtype=torch.long)
        }
        
        # 第一次尝试 - 使用model()**计算带梯度的输出
        first_attempt_outputs = self.model(**first_batch)
        
        # 获取参考模型的输出 - 不需要梯度
        with torch.no_grad():
            first_ref_outputs = self.model(**first_batch)
        
        # 计算第一次尝试的KL散度
        first_kl_loss = self.compute_kl_divergence(first_attempt_outputs.logits, first_ref_outputs.logits)
        
        # 生成第一次尝试的结果，用于评估奖励
        with torch.no_grad():  # 这一步只是为了评估，不需要梯度
            first_gen_outputs = self.model.generate(
                input_ids=batch['input_ids'].unsqueeze(0).to(device),
                attention_mask=batch['attention_mask'].unsqueeze(0).to(device),
                position_ids=batch['position_ids'].unsqueeze(0).to(device),
                pixel_values=batch['pixel_values'].to(torch.bfloat16).to(device),
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        first_attempt_responses = self.tokenizer.batch_decode(first_gen_outputs.sequences, skip_special_tokens=True)
        
        # 计算第一次尝试的奖励
        first_attempt_rewards = self.reward_function(answers, first_attempt_responses)
        
        # 准备第二次尝试输入
        second_attempt_prompts = []
        for q, first_resp in zip(questions, first_attempt_responses):
            if isinstance(q, str) and isinstance(first_resp, str):
                prompt = f"{q}\nFirst response: {first_resp}\nPlease check again and provide a better answer:"
                second_attempt_prompts.append(prompt)
            else:
                # 处理非字符串情况
                prompt = "Please check again and provide a better answer:"
                second_attempt_prompts.append(prompt)
        
        # 处理第二次尝试的输入
        second_attempt_inputs = self.tokenizer(
            second_attempt_prompts[0] if second_attempt_prompts else "Please check again:",
            return_tensors="pt",
            truncation=True
        )
        # 对于带图token的，用这个：
        second_attempt_prompts = self.tokenizer.decode(batch['input_ids'], skip_special_tokens=False).replace("What's the object in the hand","Please check again what's the object in the hand")
        second_attempt_inputs = self.tokenizer(second_attempt_prompts, return_tensors="pt", truncation=True)
        # 将second_attempt_inputs移动到device
        second_attempt_inputs = {k: v.to(device) for k, v in second_attempt_inputs.items()}
        second_attempt_attention_mask = second_attempt_inputs['attention_mask']
        second_attempt_position_ids = second_attempt_attention_mask.long().cumsum(-1) - 1
        second_attempt_position_ids.masked_fill_(second_attempt_attention_mask == 0, 1)
        
        # 准备完整的第二次尝试batch
        second_batch = {
            'input_ids': second_attempt_inputs['input_ids'],
            'attention_mask': second_attempt_attention_mask,
            'position_ids': second_attempt_position_ids,
            'pixel_values': batch['pixel_values'].to(torch.bfloat16).to(device),
            'image_flags': torch.tensor([1] * batch['pixel_values'].size(0), dtype=torch.long)
        }
        
        # 第二次尝试 - 使用model()**计算带梯度的输出
        second_attempt_outputs = self.model(**second_batch)
        
        # 获取参考模型的输出 - 不需要梯度
        with torch.no_grad():
            second_ref_outputs = self.model(**second_batch)
        
        # 计算第二次尝试的KL散度
        second_kl_loss = self.compute_kl_divergence(second_attempt_outputs.logits, second_ref_outputs.logits)
        
        # 计算总KL散度损失
        kl_loss = first_kl_loss + second_kl_loss
        
        # 生成第二次尝试的结果，用于评估奖励
        with torch.no_grad():  # 这一步只是为了评估，不需要梯度
            second_gen_outputs = self.model.generate(
                input_ids=second_batch['input_ids'],
                attention_mask=second_batch['attention_mask'],
                position_ids=second_batch['position_ids'],
                pixel_values=batch['pixel_values'].to(torch.bfloat16).to(device),
                max_new_tokens=50,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        second_attempt_responses = self.tokenizer.batch_decode(second_gen_outputs.sequences, skip_special_tokens=True)
        
        # 计算第二次尝试的奖励
        second_attempt_rewards = self.reward_function(answers, second_attempt_responses)
        
        # 计算第一次尝试的RL损失 - 使用logits
        first_rl_loss = -torch.mean(first_attempt_outputs.logits.log_softmax(-1) * first_attempt_rewards.unsqueeze(-1).unsqueeze(-1))
        
        # 计算第二次尝试的RL损失 - 使用logits
        second_rl_loss = -torch.mean(second_attempt_outputs.logits.log_softmax(-1) * second_attempt_rewards.unsqueeze(-1).unsqueeze(-1))
        
        # 总的RL损失
        rl_loss = first_rl_loss + second_rl_loss
        
        # 按照论文中的第二阶段目标函数组合损失:
        # max_θ E[∑(i=1 to 2) r̂(y_i,y*) - β₁D_KL(π_θ(·|x_i)||π_ref(·|x_i))]
        beta_1 = self.beta1  # 使用类属性
        total_loss = rl_loss + beta_1 * kl_loss
        
        return total_loss, {
            'first_rl_loss': first_rl_loss.item(),
            'second_rl_loss': second_rl_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'first_rewards': first_attempt_rewards.mean().item(),
            'second_rewards': second_attempt_rewards.mean().item()
        }

    def train(self):
        """
        SCoRe训练主函数，包括Stage I和Stage II
        适用于单样本batch的情况
        """
        print("Starting SCoRe training...")
    
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # 移动模型到对应的 GPU
        self.model = self.model.to(device)
        self.ref_model = self.ref_model.to(device)
        
        # 设置参考模型为评估模式，因为它不需要训练
        self.ref_model.eval()
        
        # 检查设备
        model_device = next(self.model.parameters()).device
        print(f"Model device after moving: {model_device}")
        
        # 配置优化器和学习率调度器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # ====== Stage I 训练 ======
        print("Starting Stage I: Training for decoupling attempts")
        self.model.train()
        stage_1_stats = {
            'total_loss': 0.0, 
            'rl_loss': 0.0, 
            'kl_loss': 0.0, 
            'first_rewards': 0.0,
            'second_rewards': 0.0
        }
        
        for epoch in range(self.stage_1_epoch):
            print(f"Stage I - Epoch {epoch+1}/{self.stage_1_epoch}")
            epoch_stats = {k: 0.0 for k in stage_1_stats.keys()}
            
            tbar = tqdm(self.train_dataset, desc=f"Stage I Training - Epoch {epoch+1}")
            for batch_idx, batch in enumerate(tbar):
                # 清除梯度
                optimizer.zero_grad()
                
                # 计算Stage I的损失 (单样本batch)
                loss, stats = self.train_step_stage_one(batch)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 参数更新
                optimizer.step()
                
                # 更新统计信息
                for k, v in stats.items():
                    epoch_stats[k] += v
                
                # 更新进度条
                tbar.set_postfix({
                    'loss': stats['total_loss'],
                    'rl_loss': stats['rl_loss'],
                    'kl_loss': stats['kl_loss'],
                    'first_reward': stats['first_rewards'],
                    'second_reward': stats['second_rewards']
                })
                
                # 每100个批次保存一次checkpoint
                if batch_idx % 100 == 0 and batch_idx > 0:
                    self.save_checkpoint(f"stage1_epoch{epoch+1}_batch{batch_idx}")
            
            # 计算平均统计信息
            batch_count = len(self.train_dataset)
            for k in epoch_stats.keys():
                epoch_stats[k] /= batch_count
                stage_1_stats[k] += epoch_stats[k]
            
            # 输出当前epoch统计信息
            print(f"Stage I - Epoch {epoch+1} stats:")
            for k, v in epoch_stats.items():
                print(f"  {k}: {v:.4f}")
            
            # 保存epoch checkpoint
            #self.save_checkpoint(f"stage1_epoch{epoch+1}")
        
        # 计算Stage I平均统计信息
        for k in stage_1_stats.keys():
            stage_1_stats[k] /= self.stage_1_epoch
        
        print("Stage I completed. Average stats:")
        for k, v in stage_1_stats.items():
            print(f"  {k}: {v:.4f}")
        
        # ====== Stage II 训练 ======
        print("\nStarting Stage II: Multi-Turn RL with Reward Shaping")
        stage_2_stats = {
            'total_loss': 0.0, 
            'first_rl_loss': 0.0, 
            'second_rl_loss': 0.0, 
            'kl_loss': 0.0, 
            'first_rewards': 0.0,
            'second_rewards': 0.0
        }
        
        for epoch in range(self.stage_2_epoch):
            print(f"Stage II - Epoch {epoch+1}/{self.stage_2_epoch}")
            epoch_stats = {k: 0.0 for k in stage_2_stats.keys()}
            
            tbar = tqdm(self.train_dataset, desc=f"Stage II Training - Epoch {epoch+1}")
            for batch_idx, batch in enumerate(tbar):
                # 清除梯度
                optimizer.zero_grad()
                
                # 计算Stage II的损失 (单样本batch)
                loss, stats = self.train_step_stage_two(batch)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 参数更新
                optimizer.step()
                
                # 更新统计信息
                for k, v in stats.items():
                    epoch_stats[k] += v
                
                # 更新进度条
                tbar.set_postfix({
                    'loss': stats['total_loss'],
                    'first_rl': stats['first_rl_loss'],
                    'second_rl': stats['second_rl_loss'],
                    'kl': stats['kl_loss'],
                    'first_reward': stats['first_rewards'],
                    'second_reward': stats['second_rewards']
                })
                
                # 每100个批次保存一次checkpoint
                if batch_idx % 100 == 0 and batch_idx > 0:
                    self.save_checkpoint(f"stage2_epoch{epoch+1}_batch{batch_idx}")
            
            # 计算平均统计信息
            batch_count = len(self.train_dataset)
            for k in epoch_stats.keys():
                epoch_stats[k] /= batch_count
                stage_2_stats[k] += epoch_stats[k]
            
            # 输出当前epoch统计信息
            print(f"Stage II - Epoch {epoch+1} stats:")
            for k, v in epoch_stats.items():
                print(f"  {k}: {v:.4f}")
            
            # 保存epoch checkpoint
            #self.save_checkpoint(f"stage2_epoch{epoch+1}")
        
        # 计算Stage II平均统计信息
        for k in stage_2_stats.keys():
            stage_2_stats[k] /= self.stage_2_epoch
        
        print("Stage II completed. Average stats:")
        for k, v in stage_2_stats.items():
            print(f"  {k}: {v:.4f}")
        
        # 保存最终模型
        self.save_checkpoint("score_final_model")
        
        print("SCoRe training completed successfully!")
        
        return {
            "stage_1_stats": stage_1_stats,
            "stage_2_stats": stage_2_stats
        }
    
    def save_checkpoint(self, name):
        """
        保存模型checkpoint为safetensors格式
        
        参数:
        - name: checkpoint名称
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # 仅在主进程上保存
        if local_rank == 0:
            from safetensors.torch import save_file
            
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 准备要保存的状态字典
            state_dict = self.model.state_dict()
            
            # 保存为safetensors格式
            checkpoint_path = os.path.join(checkpoint_dir, f"{name}.safetensors")
            save_file(state_dict, checkpoint_path)
            
            # 额外保存元数据（如果需要）
            metadata_path = os.path.join(checkpoint_dir, f"{name}_metadata.json")
            metadata = {
                'iteration': self.current_iteration,
                'hyperparams': {
                    'learning_rate': self.learning_rate,
                    'alpha': self.alpha,
                    'stage_1_epoch': self.stage_1_epoch,
                    'stage_2_epoch': self.stage_2_epoch
                }
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Model saved to {checkpoint_path}")
            print(f"Metadata saved to {metadata_path}")

    def compute_reinforce_loss(self, log_probs, response_ids, rewards, batch_size=1):
        """
        计算REINFORCE损失函数 - 适用于model.generate()返回的scores
        
        参数:
        - log_probs: 模型输出的对数概率，通常是model.generate()返回的scores
          这是一个元组(tuple)，每个元素对应一个生成步骤
        - response_ids: 生成的token ID序列，通常从model.generate返回的sequences获取
        - rewards: 根据生成结果计算的奖励值
        - batch_size: 批次大小，默认为1（单个样本）
        
        返回:
        - 损失值
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # 确保rewards在正确的设备上
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=device)
        elif rewards.device != device:
            rewards = rewards.to(device)
        
        # 确保rewards是适当的形状
        if rewards.dim() == 0:  # 单个标量
            rewards = rewards.unsqueeze(0)  # 转为 [1]
        
        # 处理scores元组: log_probs是一个元组，需要特殊处理
        if isinstance(log_probs, tuple):
            # 对于scores元组，我们计算每个token位置的得分，然后汇总
            token_log_probs = []
            
            # 获取生成的序列
            if hasattr(response_ids, 'sequences'):
                sequences = response_ids.sequences
            else:
                sequences = response_ids
                
            # 确保sequences在正确的设备上
            sequences = sequences.to(device)
            
            # 找出log_probs和sequences的有效长度
            # log_probs中的每个元素应该对应生成序列的一个位置
            seq_length = min(len(log_probs), sequences.size(1))
            
            # 收集每个token位置的对数概率
            for pos in range(seq_length):
                if pos < len(log_probs):
                    # 获取当前位置的logits
                    pos_logits = log_probs[pos].to(device)  # [batch_size, vocab_size]
                    
                    # 对于每个样本，获取该位置选择的token
                    batch_token_log_probs = []
                    for batch_idx in range(batch_size):
                        if batch_idx < sequences.size(0):
                            # 获取当前样本当前位置的token id
                            token_id = sequences[batch_idx, pos].item()
                            
                            # 获取该token的log概率
                            token_log_prob = pos_logits[batch_idx, token_id]
                            batch_token_log_probs.append(token_log_prob)
                    
                    # 如果这个位置有有效的对数概率，添加到列表
                    if batch_token_log_probs:
                        token_log_probs.append(torch.stack(batch_token_log_probs))
            
            # 如果没有收集到有效的对数概率，返回零损失
            if not token_log_probs:
                return torch.tensor(0.0, device=device)
                
            # 将收集到的对数概率堆叠成张量 [seq_length, batch_size]
            token_log_probs = torch.stack(token_log_probs).transpose(0, 1)  # [batch_size, seq_length]
            
            # 计算每个样本的REINFORCE损失
            losses = []
            for batch_idx in range(batch_size):
                if batch_idx < token_log_probs.size(0) and batch_idx < rewards.size(0):
                    # 计算负对数概率之和乘以奖励
                    sample_loss = -token_log_probs[batch_idx].sum() * rewards[batch_idx]
                    losses.append(sample_loss)
            
            # 如果没有有效损失，返回零损失
            if not losses:
                return torch.tensor(0.0, device=device)
                
            # 返回平均损失
            return torch.stack(losses).mean()
        
        else:
            # 原来的处理逻辑，用于非元组的log_probs
            losses = []
            for i in range(batch_size):
                # 获取当前样本的log_probs
                current_log_probs = log_probs[i] if isinstance(log_probs, list) else log_probs
                
                # 确保响应ID在正确的设备上
                if hasattr(response_ids, 'sequences'):
                    current_ids = response_ids.sequences[i].to(device)
                else:
                    current_ids = response_ids[i].to(device) if isinstance(response_ids, list) else response_ids.to(device)
                    
                if current_ids.dim() == 1:
                    current_ids = current_ids.unsqueeze(0)  # [1, seq_length]
                
                # 调整序列长度以匹配
                seq_length = min(current_log_probs.size(0), current_ids.size(1))
                current_log_probs = current_log_probs[:seq_length]  # [seq_length, vocab_size]
                current_ids = current_ids[:, :seq_length]  # [1, seq_length]
                
                # 创建padding掩码（排除pad token）
                padding_mask = (current_ids != self.tokenizer.pad_token_id).float()  # [1, seq_length]
                
                # 使用gather操作选择对应token的对数概率
                selected_log_probs = []
                for t in range(seq_length):
                    token_id = current_ids[0, t].item()
                    token_log_prob = current_log_probs[t, token_id]
                    selected_log_probs.append(token_log_prob)
                    
                selected_log_probs = torch.stack(selected_log_probs)
                
                # 应用padding掩码
                masked_log_probs = selected_log_probs * padding_mask.squeeze()  # [seq_length]
                
                # 应用奖励值
                current_reward = rewards[i].to(device)
                losses.append((-masked_log_probs.sum() * current_reward))
            
            # 计算平均损失
            if not losses:
                return torch.tensor(0.0, device=device)
            return torch.stack(losses).mean()
    
    def compute_kl_divergence(self, scores, ref_scores):
        """
        计算KL散度 - 用于约束模型输出与参考模型输出的差异
        适用于从model.generate()返回的scores格式
        
        参数:
        - scores: 当前模型生成的scores，来自model.generate的outputs.scores
          这是一个tuple，每个元素是一个张量，形状为[batch_size, vocab_size]
        - ref_scores: 参考模型生成的scores，来自ref_model.generate的outputs.scores
          这也是一个tuple，结构与scores相同
        
        返回:
        - KL散度值
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # 确保两个scores列表长度相同（取最小长度）
        min_length = min(len(scores), len(ref_scores))
        
        kl_divs = []
        for i in range(min_length):
            # 获取当前位置的logits
            curr_logits = scores[i]
            curr_ref_logits = ref_scores[i]
            
            # 确保两者在同一设备上
            if curr_logits.device != device:
                curr_logits = curr_logits.to(device)
            if curr_ref_logits.device != device:
                curr_ref_logits = curr_ref_logits.to(device)
            
            # 确保形状匹配
            if curr_logits.shape != curr_ref_logits.shape:
                # 如果batch维度不同，取第一个样本（对于单样本batch应该是一样的）
                if curr_logits.shape[0] != curr_ref_logits.shape[0]:
                    curr_logits = curr_logits[:1]  # 取第一个样本
                    curr_ref_logits = curr_ref_logits[:1]  # 取第一个样本
            
            # 计算当前位置的KL散度
            log_probs = F.log_softmax(curr_logits, dim=-1)
            ref_probs = F.softmax(curr_ref_logits, dim=-1)
            
            # 使用KL散度函数计算
            kl_div = F.kl_div(
                log_probs, 
                ref_probs, 
                reduction="batchmean", 
                log_target=False
            )
            
            kl_divs.append(kl_div)
        
        # 如果没有计算任何KL散度，返回0
        if len(kl_divs) == 0:
            return torch.tensor(0.0, device=device)
        
        # 返回平均KL散度
        return torch.stack(kl_divs).mean()
    
     
    def reward_function(self, gt_answers: List[str], generated_answers: List[str]) -> torch.Tensor:
        """
        计算奖励值 - 评估生成答案与参考答案的相似度
        适用于单样本情况，但也兼容多样本
        
        参数:
        - gt_answers: 标准答案，可以是字符串或字符串列表
        - generated_answers: 生成的答案，可以是字符串或字符串列表
        
        返回:
        - 奖励值张量
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        rewards = []
        
        # 处理输入类型
        if isinstance(gt_answers, str):
            gt_answers = [gt_answers]
        if isinstance(generated_answers, str):
            generated_answers = [generated_answers]
            
        # 如果generated_answers只有一个，但gt_answers有多个，则复制generated_answers
        if len(generated_answers) == 1 and len(gt_answers) > 1:
            generated_answers = [generated_answers[0]] * len(gt_answers)
        
        # 如果gt_answers只有一个，但generated_answers有多个，则复制gt_answers
        if len(gt_answers) == 1 and len(generated_answers) > 1:
            gt_answers = [gt_answers[0]] * len(generated_answers)
        
        for gt, gen in zip(gt_answers, generated_answers):
            # 提取和处理答案
            extracted_gt = extract_answer(gt)
            extracted_gen = extract_answer(gen)
            
            # 计算奖励值
            if extracted_gt == extracted_gen:
                reward = 1.0
            else:
                # 使用字符串相似度计算部分奖励
                similarity = string_similarity(extracted_gt, extracted_gen)
                reward = round((1e-3 + (1-1e-3) * similarity), 2)
            
            rewards.append(reward)
        
        # 将奖励值转换为张量并移至相应设备
        return torch.tensor(rewards, device=device)

def extract_answer(text: str) -> str:
    """
    从文本中提取答案 - 简单规范化
    
    参数:
    - text: 输入文本
    
    返回:
    - 规范化的答案文本
    """
    if not isinstance(text, str):
        text = str(text)
    return text.lower().strip()
    
def string_similarity(str1, str2):
    """
    计算两个字符串的相似度 (0-1范围)
    
    参数:
    - str1: 第一个字符串
    - str2: 第二个字符串
    
    返回:
    - 相似度值 (0-1)
    """
    # 确保输入是字符串
    if not isinstance(str1, str):
        str1 = str(str1)
    if not isinstance(str2, str):
        str2 = str(str2)
    
    try:
        from Levenshtein import distance
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0  # 两个字符串都为空
        return 1 - (distance(str1, str2) / max_len)
    except ImportError:
        # 备选方案：如果没有Levenshtein库
        if str1 == str2:
            return 1.0
        elif len(str1) == 0 or len(str2) == 0:
            return 0.0
        
        # 简单计算共同字符比例
        common_chars = set(str1) & set(str2)
        all_chars = set(str1) | set(str2)
        return len(common_chars) / len(all_chars) if all_chars else 1.0