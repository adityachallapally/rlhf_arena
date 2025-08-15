"""
RLHF Arena: Benchmarking frontier post-training RL methods for LLMs.

This package provides modular trainers for PPO, DPO, GRPO, Off-policy GRPO, 
GRPOVI, and RLAIF, with benchmarking across multiple preference datasets.
"""

from .ppo import PPOTrainer
from .dpo import DPOTrainer
from .grpo import GRPOTrainer
from .grpo_offpolicy import GRPOOffPolicyTrainer
from .grpo_vi import GRPOVITrainer
from .rlaif import RLAIFTrainer
from .utils import (
    load_config,
    setup_logging,
    setup_device,
    compute_kl_divergence,
    compute_rewards,
    save_checkpoint,
    load_checkpoint
)

__version__ = "0.1.0"
__author__ = "RLHF Arena Contributors"

__all__ = [
    "PPOTrainer",
    "DPOTrainer", 
    "GRPOTrainer",
    "GRPOOffPolicyTrainer",
    "GRPOVITrainer",
    "RLAIFTrainer",
    "load_config",
    "setup_logging",
    "setup_device",
    "compute_kl_divergence",
    "compute_rewards",
    "save_checkpoint",
    "load_checkpoint"
] 