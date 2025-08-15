"""
GRPO Off-Policy Trainer for RLHF.

This module implements the off-policy variant of GRPO for reinforcement learning from human feedback.
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


class GRPOOffPolicyTrainer:
    """
    GRPO Off-Policy Trainer for RLHF.
    
    Implements the off-policy variant of GRPO with support for:
    - Off-policy learning from experience replay
    - Importance sampling corrections
    - Target network updates
    - Mixed precision training
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GRPO Off-Policy Trainer.
        
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
        
        # Experience replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = config.get('replay_buffer_size', 10000)
        
        # Target network update frequency
        self.target_update_freq = config.get('target_update_freq', 1000)
        
        # Metrics tracking
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'reward_mean': [],
            'reward_std': [],
            'grpo_regularization_loss': [],
            'importance_weight': [],
            'target_update_count': []
        }
        
        self.logger.info("GRPO Off-Policy Trainer initialized successfully")
    
    def _init_models(self):
        """Initialize policy, target, and reference models."""
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
        
        # Load target model (copy of policy model)
        self.target_model = AutoModelForCausalLM.from_pretrained(
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
        
        # Freeze target and reference models
        for param in self.target_model.parameters():
            param.requires_grad = False
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Move models to device
        if self.config['hardware'].get('device') != 'auto':
            self.policy_model = self.policy_model.to(self.device)
            self.target_model = self.target_model.to(self.device)
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
    
    def add_to_replay_buffer(self, experience: Dict[str, torch.Tensor]):
        """Add experience to replay buffer."""
        self.replay_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
    
    def sample_from_replay_buffer(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Sample experiences from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return self.replay_buffer
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.logger.info("Target network updated")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single GRPO off-policy training step.
        
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
        
        # Forward pass through target model
        with torch.no_grad():
            target_outputs = self.target_model(
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
        target_log_probs = F.log_softmax(target_outputs.logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_outputs.logits, dim=-1)
        
        # Sample actions from policy
        action_probs = F.softmax(policy_outputs.logits, dim=-1)
        actions = torch.multinomial(action_probs, num_samples=1)
        
        # Compute action log probabilities
        policy_action_log_probs = torch.gather(policy_log_probs, -1, actions)
        target_action_log_probs = torch.gather(target_log_probs, -1, actions)
        
        # Compute importance weights
        importance_weights = torch.exp(policy_action_log_probs - target_action_log_probs)
        importance_weights = torch.clamp(importance_weights, 0.1, 10.0)  # Clip to prevent instability
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(policy_log_probs, reference_log_probs)
        
        # Compute GRPO loss
        grpo_config = self.config['grpo']
        
        # Policy loss with clipping and importance sampling
        ratio = torch.exp(policy_action_log_probs - reference_log_probs.gather(-1, actions))
        clip_epsilon = grpo_config['clip_epsilon']
        
        policy_loss_1 = importance_weights * ratio * rewards
        policy_loss_2 = importance_weights * torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * rewards
        
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
        self.metrics['importance_weight'].append(importance_weights.mean().item())
        
        # Update target network if needed
        if self.global_step % self.target_update_freq == 0:
            self.update_target_network()
            self.metrics['target_update_count'].append(1)
        else:
            self.metrics['target_update_count'].append(0)
        
        self.global_step += 1
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div.item(),
            'grpo_regularization_loss': grpo_regularization_loss.item(),
            'importance_weight': importance_weights.mean().item()
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
        
        self.logger.info(f"Starting GRPO Off-Policy training for {training_config['max_epochs']} epochs")
        
        for epoch in range(training_config['max_epochs']):
            self.epoch = epoch
            
            # Training loop
            train_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                if step >= training_config['max_steps']:
                    break
                
                # Add to replay buffer
                self.add_to_replay_buffer(batch)
                
                # Sample from replay buffer for training
                replay_batch = self.sample_from_replay_buffer(training_config['batch_size'])
                if replay_batch:
                    # Convert list of dicts to batched tensors
                    batched_replay = self._batch_replay_experiences(replay_batch)
                    loss_dict = self.train_step(batched_replay)
                else:
                    # Use current batch if replay buffer is empty
                    loss_dict = self.train_step(batch)
                
                train_losses.append(loss_dict)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'kl': f"{loss_dict['kl_divergence']:.4f}",
                    'reg': f"{loss_dict['grpo_regularization_loss']:.4f}",
                    'imp_w': f"{loss_dict['importance_weight']:.4f}"
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
        
        self.logger.info("GRPO Off-Policy training completed successfully")
    
    def _batch_replay_experiences(self, replay_experiences: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Convert list of replay experiences to batched tensors."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated batching logic
        
        batched = {}
        for key in replay_experiences[0].keys():
            if isinstance(replay_experiences[0][key], torch.Tensor):
                batched[key] = torch.stack([exp[key] for exp in replay_experiences])
            else:
                batched[key] = [exp[key] for exp in replay_experiences]
        
        return batched
    
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
            f"Imp W: {metrics['importance_weight']:.4f}"
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