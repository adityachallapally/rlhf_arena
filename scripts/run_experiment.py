#!/usr/bin/env python3
"""
RLHF Arena Experiment Runner

Main script for running single RLHF experiments with comprehensive logging,
checkpointing, and error handling.
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import json
import yaml
import logging
import signal
import psutil
import GPUtil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rlhf_arena import (
    PPOTrainer, DPOTrainer, GRPOTrainer, GRPOOffPolicyTrainer,
    GRPOVITrainer, RLAIFTrainer, load_config, setup_logging, setup_device
)


class ExperimentRunner:
    """Main experiment runner class with comprehensive error handling."""
    
    def __init__(self, config_path: str, output_dir: str, resume_from: Optional[str] = None):
        """Initialize experiment runner."""
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.resume_from = resume_from
        self.config = None
        self.trainer = None
        self.logger = None
        self.start_time = None
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self._load_config()
        
        # Setup hardware
        self._setup_hardware()
        
        # Initialize trainer
        self._init_trainer()
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.interrupted = True
        if self.trainer:
            self.trainer.save_checkpoint()
        sys.exit(0)
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_config = {
            'output_dir': str(self.output_dir),
            'log_level': 'INFO'
        }
        self.logger = setup_logging(log_config)
        
        # Also log to file
        log_file = self.output_dir / 'experiment.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Experiment runner initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self):
        """Load and validate configuration."""
        try:
            self.config = load_config(self.config_path)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
            # Validate required sections
            required_sections = ['model', 'training', 'dataset', 'hardware']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Save config to output directory
            config_output = self.output_dir / 'config.yaml'
            with open(config_output, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info("Configuration validated and saved")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_hardware(self):
        """Setup hardware configuration and monitoring."""
        try:
            # Setup device
            self.device = setup_device(self.config['hardware'])
            self.logger.info(f"Using device: {self.device}")
            
            # Log hardware info
            if torch.cuda.is_available():
                gpu_info = GPUtil.getGPUs()
                for gpu in gpu_info:
                    self.logger.info(f"GPU {gpu.id}: {gpu.name}, Memory: {gpu.memoryTotal}MB")
            
            # Log system info
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            self.logger.info(f"CPU cores: {cpu_count}, Memory: {memory.total // (1024**3)}GB")
            
        except Exception as e:
            self.logger.error(f"Failed to setup hardware: {e}")
            raise
    
    def _init_trainer(self):
        """Initialize the appropriate trainer based on configuration."""
        try:
            # Determine algorithm type from config or command line
            algorithm = self.config.get('algorithm', 'ppo').lower()
            
            # Initialize trainer based on algorithm
            if algorithm == 'ppo':
                self.trainer = PPOTrainer(self.config)
            elif algorithm == 'dpo':
                self.trainer = DPOTrainer(self.config)
            elif algorithm == 'grpo':
                self.trainer = GRPOTrainer(self.config)
            elif algorithm == 'grpo_offpolicy':
                self.trainer = GRPOOffPolicyTrainer(self.config)
            elif algorithm == 'grpo_vi':
                self.trainer = GRPOVITrainer(self.config)
            elif algorithm == 'rlaif':
                self.trainer = RLAIFTrainer(self.config)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self.logger.info(f"Initialized {algorithm.upper()} trainer")
            
            # Resume from checkpoint if specified
            if self.resume_from:
                self.trainer.load_checkpoint(self.resume_from)
                self.logger.info(f"Resumed from checkpoint: {self.resume_from}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize trainer: {e}")
            raise
    
    def _load_dataset(self):
        """Load and preprocess dataset."""
        try:
            dataset_config = self.config['dataset']
            dataset_name = dataset_config['name']
            
            self.logger.info(f"Loading dataset: {dataset_name}")
            
            # Load dataset based on name
            if dataset_name == 'hh':
                from datasets import load_dataset
                dataset = load_dataset("Anthropic/hh-rlhf")
            elif dataset_name == 'oasst':
                from datasets import load_dataset
                dataset = load_dataset("OpenAssistant/oasst1")
            elif dataset_name == 'ultrafeedback':
                from datasets import load_dataset
                dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            # Apply preprocessing
            dataset = self._preprocess_dataset(dataset, dataset_config)
            
            self.logger.info(f"Dataset loaded: {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _preprocess_dataset(self, dataset, config):
        """Preprocess dataset according to configuration."""
        try:
            preprocessing = config.get('preprocessing', {})
            
            # Filter by length
            if preprocessing.get('filter_by_length', False):
                min_length = preprocessing.get('min_length', 10)
                max_length = preprocessing.get('max_length', 512)
                
                def filter_length(example):
                    text = example.get('text', '')
                    if isinstance(text, str):
                        return min_length <= len(text.split()) <= max_length
                    return True
                
                dataset = dataset.filter(filter_length)
                self.logger.info(f"Filtered dataset by length: {len(dataset)} samples")
            
            # Remove duplicates
            if preprocessing.get('remove_duplicates', False):
                dataset = dataset.unique('text')
                self.logger.info(f"Removed duplicates: {len(dataset)} samples")
            
            # Limit samples
            max_samples = config.get('max_samples')
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                self.logger.info(f"Limited to {max_samples} samples")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"Failed to preprocess dataset: {e}")
            raise
    
    def _setup_monitoring(self):
        """Setup experiment monitoring and tracking."""
        try:
            logging_config = self.config.get('logging', {})
            
            # Setup Weights & Biases
            if logging_config.get('use_wandb', False):
                import wandb
                wandb.init(
                    project=logging_config.get('wandb_project', 'rlhf_arena'),
                    name=logging_config.get('wandb_run_name', 'experiment'),
                    config=self.config,
                    dir=str(self.output_dir)
                )
                self.logger.info("Weights & Biases initialized")
            
            # Setup TensorBoard
            if logging_config.get('use_tensorboard', False):
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = self.output_dir / 'tensorboard'
                tensorboard_dir.mkdir(exist_ok=True)
                self.writer = SummaryWriter(str(tensorboard_dir))
                self.logger.info("TensorBoard initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            # Continue without monitoring if it fails
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all configured outputs."""
        try:
            # Log to console
            self.logger.info(f"Step {step}: {metrics}")
            
            # Log to TensorBoard
            if hasattr(self, 'writer'):
                for key, value in metrics.items():
                    self.writer.add_scalar(key, value, step)
            
            # Log to W&B
            if wandb.run is not None:
                wandb.log(metrics, step=step)
                
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def _save_checkpoint(self, step: int):
        """Save training checkpoint."""
        try:
            checkpoint_dir = self.output_dir / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pt'
            self.trainer.save_checkpoint(str(checkpoint_path))
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _evaluate(self, step: int):
        """Run evaluation."""
        try:
            eval_config = self.config.get('evaluation', {})
            
            if step % eval_config.get('eval_steps', 200) == 0:
                self.logger.info(f"Running evaluation at step {step}")
                
                # Run evaluation
                eval_metrics = self.trainer.evaluate()
                
                # Log evaluation metrics
                self._log_metrics(eval_metrics, step)
                
                # Save best model if needed
                if eval_metrics.get('reward_mean', 0) > getattr(self, 'best_reward', float('-inf')):
                    self.best_reward = eval_metrics['reward_mean']
                    best_model_path = self.output_dir / 'best_model.pt'
                    self.trainer.save_checkpoint(str(best_model_path))
                    self.logger.info(f"New best model saved: {best_model_path}")
                
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
    
    def run(self):
        """Run the main experiment."""
        try:
            self.start_time = time.time()
            self.logger.info("Starting experiment...")
            
            # Setup monitoring
            self._setup_monitoring()
            
            # Load dataset
            dataset = self._load_dataset()
            
            # Training loop
            training_config = self.config['training']
            total_steps = training_config.get('total_steps', 1000)
            save_steps = training_config.get('save_steps', 100)
            logging_steps = training_config.get('logging_steps', 10)
            
            self.logger.info(f"Training for {total_steps} steps")
            
            for step in range(total_steps):
                if self.interrupted:
                    break
                
                # Training step
                metrics = self.trainer.train_step(dataset)
                
                # Log metrics
                if step % logging_steps == 0:
                    self._log_metrics(metrics, step)
                
                # Save checkpoint
                if step % save_steps == 0:
                    self._save_checkpoint(step)
                
                # Evaluate
                self._evaluate(step)
                
                # Check for early stopping
                if metrics.get('reward_mean', 0) > training_config.get('target_reward', float('inf')):
                    self.logger.info(f"Target reward achieved at step {step}")
                    break
            
            # Final evaluation and checkpoint
            self.logger.info("Training completed, running final evaluation...")
            final_metrics = self.trainer.evaluate()
            self._log_metrics(final_metrics, total_steps)
            
            # Save final model
            final_model_path = self.output_dir / 'final_model.pt'
            self.trainer.save_checkpoint(str(final_model_path))
            
            # Log experiment summary
            self._log_experiment_summary()
            
            self.logger.info("Experiment completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _log_experiment_summary(self):
        """Log experiment summary and statistics."""
        try:
            end_time = time.time()
            duration = end_time - self.start_time
            
            summary = {
                'experiment_duration_seconds': duration,
                'experiment_duration_hours': duration / 3600,
                'final_metrics': getattr(self.trainer, 'metrics', {}),
                'best_reward': getattr(self, 'best_reward', None),
                'config_path': self.config_path,
                'output_dir': str(self.output_dir)
            }
            
            # Save summary
            summary_path = self.output_dir / 'experiment_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info("Experiment summary saved")
            self.logger.info(f"Total duration: {duration/3600:.2f} hours")
            
        except Exception as e:
            self.logger.error(f"Failed to save experiment summary: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RLHF Arena Experiment Runner")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--resume_from", help="Resume from checkpoint")
    parser.add_argument("--algorithm", help="Override algorithm from config")
    parser.add_argument("--dataset", help="Override dataset from config")
    
    args = parser.parse_args()
    
    try:
        # Create and run experiment
        runner = ExperimentRunner(
            config_path=args.config,
            output_dir=args.output_dir,
            resume_from=args.resume_from
        )
        
        # Override config if specified
        if args.algorithm:
            runner.config['algorithm'] = args.algorithm
        if args.dataset:
            runner.config['dataset']['name'] = args.dataset
        
        runner.run()
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
