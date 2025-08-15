"""
Utility functions for RLHF Arena.

This module provides common utilities used across all trainer classes.
"""

import torch
import torch.nn.functional as F
import yaml
import logging
import os
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(logging_config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        logging_config: Logging configuration dictionary
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger('rlhf_arena')
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Setup file logging if specified
        if logging_config.get('output_dir'):
            os.makedirs(logging_config['output_dir'], exist_ok=True)
            log_file = os.path.join(logging_config['output_dir'], 'training.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def setup_device(hardware_config: Dict[str, Any]) -> torch.device:
    """
    Setup device configuration.
    
    Args:
        hardware_config: Hardware configuration dictionary
        
    Returns:
        PyTorch device
    """
    device_type = hardware_config.get('device', 'auto')
    
    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_type)
    
    return device


def compute_kl_divergence(log_probs_1: torch.Tensor, log_probs_2: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two log probability distributions.
    
    Args:
        log_probs_1: First log probability distribution
        log_probs_2: Second log probability distribution
        
    Returns:
        KL divergence
    """
    # Convert log probabilities to probabilities
    probs_1 = F.softmax(log_probs_1, dim=-1)
    probs_2 = F.softmax(log_probs_2, dim=-1)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    probs_1 = probs_1 + eps
    probs_2 = probs_2 + eps
    
    # Normalize
    probs_1 = probs_1 / probs_1.sum(dim=-1, keepdim=True)
    probs_2 = probs_2 / probs_2.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence: KL(p1 || p2) = sum(p1 * log(p1/p2))
    kl_div = (probs_1 * (torch.log(probs_1) - torch.log(probs_2))).sum(dim=-1)
    
    return kl_div.mean()


def compute_rewards(
    responses: torch.Tensor,
    reward_model: Optional[torch.nn.Module] = None,
    reward_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Compute rewards for responses.
    
    Args:
        responses: Model responses
        reward_model: Optional reward model
        reward_weights: Optional reward weights for multi-objective
        
    Returns:
        Reward tensor
    """
    if reward_model is not None:
        # Use reward model to compute rewards
        with torch.no_grad():
            rewards = reward_model(responses)
    else:
        # Placeholder rewards (random for now)
        batch_size = responses.shape[0] if len(responses.shape) > 0 else 1
        rewards = torch.randn(batch_size, device=responses.device)
    
    # Apply reward weights if specified
    if reward_weights:
        # This is a simplified implementation
        # In practice, you'd compute different reward components
        total_weight = sum(reward_weights.values())
        if total_weight > 0:
            rewards = rewards * (total_weight / len(reward_weights))
    
    return rewards


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    epoch: int,
    best_reward: float,
    checkpoint_path: str
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        global_step: Current global step
        epoch: Current epoch
        best_reward: Best reward achieved
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'global_step': global_step,
        'epoch': epoch,
        'best_reward': best_reward,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """
    Create a PyTorch DataLoader.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def setup_mixed_precision(hardware_config: Dict[str, Any]):
    """
    Setup mixed precision training.
    
    Args:
        hardware_config: Hardware configuration dictionary
    """
    mixed_precision = hardware_config.get('mixed_precision', None)
    
    if mixed_precision == 'fp16':
        try:
            from apex import amp
            return amp
        except ImportError:
            print("Warning: Apex not available, falling back to torch.cuda.amp")
            return torch.cuda.amp
    elif mixed_precision == 'bf16':
        return torch.cuda.amp
    else:
        return None


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metric_names: List of metric names to compute
        
    Returns:
        Dictionary of metric values
    """
    if metric_names is None:
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
    
    metrics = {}
    
    if 'accuracy' in metric_names:
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=-1)
        else:
            pred_labels = predictions
        accuracy = (pred_labels == targets).float().mean().item()
        metrics['accuracy'] = accuracy
    
    if 'precision' in metric_names:
        # Simplified precision calculation
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=-1)
        else:
            pred_labels = predictions
        precision = ((pred_labels == 1) & (targets == 1)).float().sum() / max((pred_labels == 1).float().sum(), 1)
        metrics['precision'] = precision.item()
    
    if 'recall' in metric_names:
        # Simplified recall calculation
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=-1)
        else:
            pred_labels = predictions
        recall = ((pred_labels == 1) & (targets == 1)).float().sum() / max((targets == 1).float().sum(), 1)
        metrics['recall'] = recall.item()
    
    if 'f1' in metric_names:
        if 'precision' in metrics and 'recall' in metrics:
            precision = metrics['precision']
            recall = metrics['recall']
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            metrics['f1'] = f1
    
    return metrics


def setup_wandb(logging_config: Dict[str, Any], config: Dict[str, Any]):
    """
    Setup Weights & Biases logging.
    
    Args:
        logging_config: Logging configuration
        config: Full configuration dictionary
    """
    if logging_config.get('log_to_wandb', False):
        try:
            import wandb
            
            wandb.init(
                project=logging_config.get('project_name', 'rlhf-arena'),
                name=logging_config.get('run_name', 'experiment'),
                config=config
            )
            
            return wandb
        except ImportError:
            print("Warning: wandb not available, skipping W&B logging")
            return None
    return None


def setup_tensorboard(logging_config: Dict[str, Any]):
    """
    Setup TensorBoard logging.
    
    Args:
        logging_config: Logging configuration
    """
    if logging_config.get('log_to_tensorboard', False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = logging_config.get('output_dir', './logs')
            os.makedirs(log_dir, exist_ok=True)
            
            writer = SummaryWriter(log_dir=log_dir)
            return writer
        except ImportError:
            print("Warning: tensorboard not available, skipping TensorBoard logging")
            return None
    return None


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    wandb_logger=None,
    tensorboard_writer=None
):
    """
    Log metrics to various logging backends.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step
        wandb_logger: Weights & Biases logger
        tensorboard_writer: TensorBoard writer
    """
    # Log to W&B
    if wandb_logger is not None:
        wandb_logger.log(metrics, step=step)
    
    # Log to TensorBoard
    if tensorboard_writer is not None:
        for name, value in metrics.items():
            tensorboard_writer.add_scalar(name, value, step)


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to experiment directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    return experiment_dir


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_pretrained_model(model_name: str, device: torch.device = None):
    """
    Load a pretrained model.
    
    Args:
        model_name: Name of the pretrained model
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if device is not None:
        model = model.to(device)
    
    return model, tokenizer


def setup_gradient_checkpointing(model: torch.nn.Module, enabled: bool = True):
    """
    Setup gradient checkpointing for a model.
    
    Args:
        model: Model to setup gradient checkpointing for
        enabled: Whether to enable gradient checkpointing
    """
    if enabled:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()


def compute_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """
    Compute model size statistics.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model size statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def setup_distributed_training(hardware_config: Dict[str, Any]):
    """
    Setup distributed training.
    
    Args:
        hardware_config: Hardware configuration dictionary
    """
    if hardware_config.get('ddp', False):
        try:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            # Initialize distributed training
            dist.init_process_group(backend='nccl')
            
            return dist, DDP
        except ImportError:
            print("Warning: Distributed training not available")
            return None, None
    return None, None


def cleanup_distributed_training():
    """
    Cleanup distributed training.
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass 