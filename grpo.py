"""
GRPO (Generalized Relative Policy Optimization) Trainer for RLHF.

This module implements the GRPO algorithm for reinforcement learning from human feedback.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import os

from .utils import setup_logging, setup_device, compute_kl_divergence, save_checkpoint, load_checkpoint


class GRPOTrainer:
    """
    GRPO Trainer for RLHF.
    
    Implements the GRPO algorithm with support for:
    - Generalized policy optimization
    - Regularization techniques
    - Advantage normalization
    - Mixed precision training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GRPO Trainer.
        
        Args:
            config: Configuration dictionary containing all hyperparameters
        """
        self.config = config
        self.device = setup_device(config.get('hardware', {}))
        
        # Setup logging
        self.logger = setup_logging(config.get('logging', {}))
        
        # Initialize models
        self._init_models()
        self._init_optimizer()
        self._init_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')
        
        # Metrics tracking
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'reward_mean': [],
            'reward_std': [],
            'grpo_regularization_loss': [],
            'advantage_norm': [],
            'clip_fraction': []
        }
        
        self.logger.info("GRPO Trainer initialized successfully")
    
    def _init_models(self):
        """Initialize policy and reference models."""
        model_config = self.config['model']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['checkpoint'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_config['checkpoint'],
            torch_dtype=torch.float16 if self.config['hardware'].get('mixed_precision') == 'fp16' else torch.float32,
            device_map='auto' if self.config['hardware'].get('device') == 'auto' else None
        )
        
        # Load reference model for KL computation
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            model_config['checkpoint'],
            torch_dtype=torch.float16 if self.config['hardware'].get('mixed_precision') == 'fp16' else torch.float32,
            device_map='auto' if self.config['hardware'].get('device') == 'auto' else None
        )
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Move models to device
        if self.config['hardware'].get('device') != 'auto':
            self.policy_model = self.policy_model.to(self.device)
            self.reference_model = self.reference_model.to(self.device)
        
        # Enable gradient checkpointing if specified
        if self.config['hardware'].get('gradient_checkpointing', False):
            self.policy_model.gradient_checkpointing_enable()
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        training_config = self.config['training']
        
        # Get parameter names for weight decay
        decay_parameters = get_parameter_names(self.policy_model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.policy_model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": training_config.get('weight_decay', 0.01),
            },
            {
                "params": [p for n, p in self.policy_model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        training_config = self.config['training']
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config['max_steps'],
            eta_min=training_config['learning_rate'] * 0.1
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single GRPO training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary containing loss values and metrics
        """
        self.policy_model.train()
        
        # Extract batch data
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        
        # Forward pass through policy model
        policy_outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Forward pass through reference model
        with torch.no_grad():
            reference_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Compute log probabilities
        policy_log_probs = F.log_softmax(policy_outputs.logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_outputs.logits, dim=-1)
        
        # Sample actions from policy
        action_probs = F.softmax(policy_outputs.logits, dim=-1)
        actions = torch.multinomial(action_probs, num_samples=1)
        
        # Compute action log probabilities
        action_log_probs = torch.gather(policy_log_probs, -1, actions)
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(policy_log_probs, reference_log_probs)
        
        # Compute GRPO loss
        grpo_config = self.config['grpo']
        
        # Policy loss with clipping
        ratio = torch.exp(action_log_probs - reference_log_probs.gather(-1, actions))
        clip_epsilon = grpo_config['clip_epsilon']
        
        policy_loss_1 = ratio * rewards
        policy_loss_2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * rewards
        
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value loss (placeholder - would need value function)
        value_loss = torch.tensor(0.0, device=self.device)
        
        # Entropy bonus
        entropy = -(action_probs * policy_log_probs).sum(-1).mean()
        entropy_loss = -grpo_config['entropy_coef'] * entropy
        
        # KL penalty
        kl_loss = grpo_config['beta'] * kl_div
        
        # GRPO regularization loss
        alpha = grpo_config['alpha']
        grpo_regularization_loss = alpha * self._compute_regularization_loss(policy_log_probs, reference_log_probs)
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss + kl_loss + grpo_regularization_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), 
            grpo_config['max_grad_norm']
        )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        self.metrics['policy_loss'].append(policy_loss.item())
        self.metrics['value_loss'].append(value_loss.item())
        self.metrics['entropy'].append(entropy.item())
        self.metrics['kl_divergence'].append(kl_div.item())
        self.metrics['reward_mean'].append(rewards.mean().item())
        self.metrics['reward_std'].append(rewards.std().item())
        self.metrics['grpo_regularization_loss'].append(grpo_regularization_loss.item())
        
        # Compute clip fraction
        clip_fraction = (torch.abs(ratio - 1) > clip_epsilon).float().mean().item()
        self.metrics['clip_fraction'].append(clip_fraction)
        
        # Compute advantage norm
        advantage_norm = torch.norm(rewards).item()
        self.metrics['advantage_norm'].append(advantage_norm)
        
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div.item(),
            'grpo_regularization_loss': grpo_regularization_loss.item(),
            'clip_fraction': clip_fraction,
            'advantage_norm': advantage_norm
        }
    
    def _compute_regularization_loss(self, policy_log_probs: torch.Tensor, reference_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute GRPO regularization loss."""
        # This is a placeholder for the actual regularization computation
        # In practice, this could be various forms of regularization
        
        # Example: L2 regularization on log probability differences
        log_prob_diff = policy_log_probs - reference_log_probs
        regularization_loss = torch.norm(log_prob_diff, p=2, dim=-1).mean()
        
        return regularization_loss
    
    def evaluate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform evaluation step.
        
        Args:
            batch: Batch of evaluation data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.policy_model.eval()
        
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            
            # Forward pass
            policy_outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            reference_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Compute metrics
            policy_log_probs = F.log_softmax(policy_outputs.logits, dim=-1)
            reference_log_probs = F.log_softmax(reference_outputs.logits, dim=-1)
            
            kl_div = compute_kl_divergence(policy_log_probs, reference_log_probs)
            
            return {
                'kl_divergence': kl_div.item(),
                'reward_mean': rewards.mean().item(),
                'reward_std': rewards.std().item()
            }
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
        """
        training_config = self.config['training']
        
        self.logger.info(f"Starting GRPO training for {training_config['max_epochs']} epochs")
        
        for epoch in range(training_config['max_epochs']):
            self.epoch = epoch
            
            # Training loop
            train_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                if step >= training_config['max_steps']:
                    break
                
                loss_dict = self.train_step(batch)
                train_losses.append(loss_dict)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'kl': f"{loss_dict['kl_divergence']:.4f}",
                    'reg': f"{loss_dict['grpo_regularization_loss']:.4f}",
                    'clip': f"{loss_dict['clip_fraction']:.4f}"
                })
                
                # Logging
                if self.global_step % training_config['logging_steps'] == 0:
                    self._log_metrics(loss_dict)
                
                # Evaluation
                if eval_dataloader and self.global_step % training_config['eval_steps'] == 0:
                    self._evaluate(eval_dataloader)
                
                # Save checkpoint
                if self.global_step % training_config['save_steps'] == 0:
                    self._save_checkpoint()
            
            # End of epoch evaluation
            if eval_dataloader:
                self._evaluate(eval_dataloader)
            
            # Save checkpoint at end of epoch
            self._save_checkpoint()
        
        self.logger.info("GRPO training completed successfully")
    
    def _evaluate(self, eval_dataloader: DataLoader):
        """Run evaluation on the evaluation dataset."""
        self.policy_model.eval()
        
        eval_metrics = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                metrics = self.evaluate_step(batch)
                eval_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = {}
        for key in eval_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in eval_metrics])
        
        self.logger.info(f"Evaluation metrics: {avg_metrics}")
        
        # Update best reward
        if avg_metrics['reward_mean'] > self.best_reward:
            self.best_reward = avg_metrics['reward_mean']
            self._save_checkpoint(is_best=True)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        self.logger.info(
            f"Step {self.global_step}: "
            f"Loss: {metrics['total_loss']:.4f}, "
            f"KL: {metrics['kl_divergence']:.4f}, "
            f"Reg: {metrics['grpo_regularization_loss']:.4f}, "
            f"Clip: {metrics['clip_fraction']:.4f}"
        )
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config['logging']['output_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")
        
        save_checkpoint(
            self.policy_model,
            self.optimizer,
            self.scheduler,
            self.global_step,
            self.epoch,
            self.best_reward,
            checkpoint_path
        )
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(
                self.policy_model,
                self.optimizer,
                self.scheduler,
                self.global_step,
                self.epoch,
                self.best_reward,
                best_path
            )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path)
        
        self.policy_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_reward = checkpoint['best_reward']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics."""
        return self.metrics.copy() 