"""
DPO (Direct Preference Optimization) Trainer for RLHF.

This module implements the DPO algorithm for reinforcement learning from human feedback.
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


class DPOTrainer:
    """
    DPO Trainer for RLHF.
    
    Implements the DPO algorithm with support for:
    - Preference learning from human feedback
    - KL divergence regularization
    - Reference model freezing
    - Mixed precision training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DPO Trainer.
        
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
            'dpo_loss': [],
            'reference_loss': [],
            'kl_divergence': [],
            'chosen_rewards': [],
            'rejected_rewards': [],
            'reward_diff': [],
            'policy_accuracy': []
        }
        
        self.logger.info("DPO Trainer initialized successfully")
    
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
        Perform a single DPO training step.
        
        Args:
            batch: Batch of training data with chosen and rejected responses
            
        Returns:
            Dictionary containing loss values and metrics
        """
        self.policy_model.train()
        
        # Extract batch data
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        # Forward pass through policy model
        chosen_outputs = self.policy_model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            return_dict=True
        )
        
        rejected_outputs = self.policy_model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            return_dict=True
        )
        
        # Forward pass through reference model
        with torch.no_grad():
            chosen_ref_outputs = self.reference_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                return_dict=True
            )
            
            rejected_ref_outputs = self.reference_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                return_dict=True
            )
        
        # Compute log probabilities
        chosen_log_probs = self._get_log_probs(chosen_outputs.logits, chosen_input_ids)
        rejected_log_probs = self._get_log_probs(rejected_outputs.logits, rejected_input_ids)
        
        chosen_ref_log_probs = self._get_log_probs(chosen_ref_outputs.logits, chosen_input_ids)
        rejected_ref_log_probs = self._get_log_probs(rejected_ref_outputs.logits, rejected_input_ids)
        
        # Compute DPO loss
        dpo_config = self.config['dpo']
        beta = dpo_config['beta']
        
        # Compute log ratios
        chosen_log_ratio = chosen_log_probs - chosen_ref_log_probs
        rejected_log_ratio = rejected_log_probs - rejected_ref_log_probs
        
        # Compute DPO loss
        if dpo_config['loss_type'] == 'sigmoid':
            # Sigmoid loss
            losses = -F.logsigmoid(beta * (chosen_log_ratio - rejected_log_ratio))
        else:
            # Hinge loss
            losses = torch.relu(1 - beta * (chosen_log_ratio - rejected_log_ratio))
        
        dpo_loss = losses.mean()
        
        # Compute reference loss (optional)
        reference_loss = torch.tensor(0.0, device=self.device)
        if dpo_config.get('reference_free', False):
            # Reference-free variant
            pass
        
        # Compute KL divergence
        kl_div_chosen = compute_kl_divergence(
            F.log_softmax(chosen_outputs.logits, dim=-1),
            F.log_softmax(chosen_ref_outputs.logits, dim=-1)
        )
        
        kl_div_rejected = compute_kl_divergence(
            F.log_softmax(rejected_outputs.logits, dim=-1),
            F.log_softmax(rejected_ref_outputs.logits, dim=-1)
        )
        
        kl_div = (kl_div_chosen + kl_div_rejected) / 2
        
        # Total loss
        total_loss = dpo_loss + reference_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        self.metrics['dpo_loss'].append(dpo_loss.item())
        self.metrics['reference_loss'].append(reference_loss.item())
        self.metrics['kl_divergence'].append(kl_div.item())
        self.metrics['chosen_rewards'].append(chosen_log_ratio.mean().item())
        self.metrics['rejected_rewards'].append(rejected_log_ratio.mean().item())
        self.metrics['reward_diff'].append((chosen_log_ratio - rejected_log_ratio).mean().item())
        
        # Compute policy accuracy (how often chosen > rejected)
        policy_accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
        self.metrics['policy_accuracy'].append(policy_accuracy)
        
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'dpo_loss': dpo_loss.item(),
            'reference_loss': reference_loss.item(),
            'kl_divergence': kl_div.item(),
            'policy_accuracy': policy_accuracy,
            'reward_diff': (chosen_log_ratio - rejected_log_ratio).mean().item()
        }
    
    def _get_log_probs(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for the target tokens."""
        # Shift logits and input_ids to align
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for target tokens
        target_log_probs = torch.gather(log_probs, -1, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding tokens
        mask = (target_ids != self.tokenizer.pad_token_id).float()
        
        # Return mean log prob per sequence
        return (target_log_probs * mask).sum(-1) / mask.sum(-1)
    
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
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            
            # Forward pass
            chosen_outputs = self.policy_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                return_dict=True
            )
            
            rejected_outputs = self.policy_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                return_dict=True
            )
            
            chosen_ref_outputs = self.reference_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                return_dict=True
            )
            
            rejected_ref_outputs = self.reference_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                return_dict=True
            )
            
            # Compute metrics
            chosen_log_probs = self._get_log_probs(chosen_outputs.logits, chosen_input_ids)
            rejected_log_probs = self._get_log_probs(rejected_outputs.logits, rejected_input_ids)
            
            chosen_ref_log_probs = self._get_log_probs(chosen_ref_outputs.logits, chosen_input_ids)
            rejected_ref_log_probs = self._get_log_probs(rejected_ref_outputs.logits, rejected_input_ids)
            
            # Compute KL divergence
            kl_div_chosen = compute_kl_divergence(
                F.log_softmax(chosen_outputs.logits, dim=-1),
                F.log_softmax(chosen_ref_outputs.logits, dim=-1)
            )
            
            kl_div_rejected = compute_kl_divergence(
                F.log_softmax(rejected_outputs.logits, dim=-1),
                F.log_softmax(rejected_ref_outputs.logits, dim=-1)
            )
            
            kl_div = (kl_div_chosen + kl_div_rejected) / 2
            
            # Compute policy accuracy
            chosen_log_ratio = chosen_log_probs - chosen_ref_log_probs
            rejected_log_ratio = rejected_log_probs - rejected_ref_log_probs
            policy_accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
            
            return {
                'kl_divergence': kl_div.item(),
                'policy_accuracy': policy_accuracy,
                'chosen_rewards': chosen_log_ratio.mean().item(),
                'rejected_rewards': rejected_log_ratio.mean().item(),
                'reward_diff': (chosen_log_ratio - rejected_log_ratio).mean().item()
            }
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
        """
        training_config = self.config['training']
        
        self.logger.info(f"Starting DPO training for {training_config['max_epochs']} epochs")
        
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
                    'dpo_loss': f"{loss_dict['dpo_loss']:.4f}",
                    'accuracy': f"{loss_dict['policy_accuracy']:.4f}"
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
        
        self.logger.info("DPO training completed successfully")
    
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
        if avg_metrics['policy_accuracy'] > self.best_reward:
            self.best_reward = avg_metrics['policy_accuracy']
            self._save_checkpoint(is_best=True)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        self.logger.info(
            f"Step {self.global_step}: "
            f"Loss: {metrics['total_loss']:.4f}, "
            f"DPO Loss: {metrics['dpo_loss']:.4f}, "
            f"Accuracy: {metrics['policy_accuracy']:.4f}"
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