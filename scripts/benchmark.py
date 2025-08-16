#!/usr/bin/env python3
"""
RLHF Arena Benchmark Orchestrator

Multi-experiment orchestrator for comprehensive algorithm benchmarking
with automated experiment management, progress tracking, and result aggregation.
"""

import argparse
import os
import sys
import time
import traceback
import subprocess
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import psutil
import GPUtil
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rlhf_arena import load_config, setup_logging


class BenchmarkOrchestrator:
    """Multi-experiment benchmark orchestrator."""
    
    def __init__(self, config_path: str, output_dir: str, max_parallel: int = 1):
        """Initialize benchmark orchestrator."""
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.max_parallel = max_parallel
        self.config = None
        self.logger = None
        self.experiments = []
        self.running_experiments = {}
        self.completed_experiments = []
        self.failed_experiments = []
        self.start_time = None
        self.interrupted = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self._load_config()
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.interrupted = True
        self._cleanup_running_experiments()
        sys.exit(0)
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_config = {
            'output_dir': str(self.output_dir),
            'log_level': 'INFO'
        }
        self.logger = setup_logging(log_config)
        
        # Also log to file
        log_file = self.output_dir / 'benchmark.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Benchmark orchestrator initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Max parallel experiments: {self.max_parallel}")
    
    def _load_config(self):
        """Load and validate benchmark configuration."""
        try:
            self.config = load_config(self.config_path)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
            # Validate required sections
            required_sections = ['benchmark', 'algorithms', 'datasets']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Save config to output directory
            config_output = self.output_dir / 'benchmark_config.yaml'
            with open(config_output, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info("Benchmark configuration validated and saved")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking and state management."""
        try:
            # Load existing state if available
            state_file = self.output_dir / 'benchmark_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.completed_experiments = state.get('completed', [])
                    self.failed_experiments = state.get('failed', [])
                    self.logger.info(f"Loaded existing state: {len(self.completed_experiments)} completed, {len(self.failed_experiments)} failed")
            
            # Create experiment tracking file
            self.tracking_file = self.output_dir / 'experiment_tracking.json'
            
            # Initialize tracking
            self._save_tracking_state()
            
        except Exception as e:
            self.logger.error(f"Failed to setup experiment tracking: {e}")
            raise
    
    def _generate_experiments(self):
        """Generate all experiments to run based on configuration."""
        try:
            algorithms = self.config['algorithms']
            datasets = self.config['datasets']
            model_sizes = self.config.get('model_sizes', ['7b'])
            
            experiments = []
            
            for algorithm in algorithms:
                for dataset in datasets:
                    for model_size in model_sizes:
                        # Create experiment configuration
                        exp_config = self._create_experiment_config(algorithm, dataset, model_size)
                        
                        # Create experiment metadata
                        exp_meta = {
                            'id': f"{algorithm}_{dataset}_{model_size}_{int(time.time())}",
                            'algorithm': algorithm,
                            'dataset': dataset,
                            'model_size': model_size,
                            'config': exp_config,
                            'status': 'pending',
                            'created_at': datetime.now().isoformat(),
                            'started_at': None,
                            'completed_at': None,
                            'duration': None,
                            'output_dir': str(self.output_dir / f"{algorithm}_{dataset}_{model_size}"),
                            'checkpoint_dir': str(self.output_dir / f"{algorithm}_{dataset}_{model_size}" / 'checkpoints'),
                            'log_file': str(self.output_dir / f"{algorithm}_{dataset}_{model_size}" / 'experiment.log')
                        }
                        
                        experiments.append(exp_meta)
            
            self.experiments = experiments
            self.logger.info(f"Generated {len(experiments)} experiments")
            
            # Save experiment list
            experiments_file = self.output_dir / 'experiments.json'
            with open(experiments_file, 'w') as f:
                json.dump(experiments, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to generate experiments: {e}")
            raise
    
    def _create_experiment_config(self, algorithm: str, dataset: str, model_size: str) -> Dict[str, Any]:
        """Create experiment-specific configuration."""
        try:
            # Load base algorithm config
            base_config_path = Path(__file__).parent.parent / 'configs' / f'{algorithm}.yaml'
            if not base_config_path.exists():
                raise ValueError(f"Base config not found: {base_config_path}")
            
            with open(base_config_path, 'r') as f:
                base_config = yaml.safe_load(f)
            
            # Override with benchmark-specific settings
            base_config['algorithm'] = algorithm
            base_config['dataset']['name'] = dataset
            
            # Model size specific overrides
            if model_size == '7b':
                base_config['model']['checkpoint'] = "microsoft/DialoGPT-medium"
                base_config['training']['batch_size'] = 4
            elif model_size == '13b':
                base_config['model']['checkpoint'] = "microsoft/DialoGPT-large"
                base_config['training']['batch_size'] = 2
            elif model_size == '30b':
                base_config['model']['checkpoint'] = "microsoft/DialoGPT-xlarge"
                base_config['training']['batch_size'] = 1
            
            # Benchmark-specific overrides
            benchmark_config = self.config.get('experiment_overrides', {})
            for key, value in benchmark_config.items():
                keys = key.split('.')
                config = base_config
                for k in keys[:-1]:
                    config = config.setdefault(k, {})
                config[keys[-1]] = value
            
            return base_config
            
        except Exception as e:
            self.logger.error(f"Failed to create experiment config: {e}")
            raise
    
    def _run_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment."""
        try:
            exp_id = experiment['id']
            self.logger.info(f"Starting experiment: {exp_id}")
            
            # Update experiment status
            experiment['status'] = 'running'
            experiment['started_at'] = datetime.now().isoformat()
            self._save_tracking_state()
            
            # Create experiment output directory
            exp_output_dir = Path(experiment['output_dir'])
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save experiment config
            config_file = exp_output_dir / 'experiment_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(experiment['config'], f, default_flow_style=False)
            
            # Run experiment using subprocess
            cmd = [
                sys.executable, str(Path(__file__).parent / 'run_experiment.py'),
                '--config', str(config_file),
                '--output_dir', str(exp_output_dir)
            ]
            
            # Run experiment
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.get('experiment_timeout', 3600)  # 1 hour default
            )
            duration = time.time() - start_time
            
            # Process results
            if result.returncode == 0:
                experiment['status'] = 'completed'
                experiment['duration'] = duration
                experiment['completed_at'] = datetime.now().isoformat()
                experiment['stdout'] = result.stdout
                experiment['stderr'] = result.stderr
                
                self.logger.info(f"Experiment {exp_id} completed successfully in {duration/3600:.2f} hours")
                
                # Parse metrics from output
                metrics = self._parse_experiment_metrics(exp_output_dir)
                experiment['metrics'] = metrics
                
            else:
                experiment['status'] = 'failed'
                experiment['duration'] = duration
                experiment['completed_at'] = datetime.now().isoformat()
                experiment['stdout'] = result.stdout
                experiment['stderr'] = result.stderr
                experiment['return_code'] = result.returncode
                
                self.logger.error(f"Experiment {exp_id} failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
            
            return experiment
            
        except subprocess.TimeoutExpired:
            experiment['status'] = 'timeout'
            experiment['duration'] = self.config.get('experiment_timeout', 3600)
            experiment['completed_at'] = datetime.now().isoformat()
            experiment['error'] = 'Experiment timed out'
            
            self.logger.error(f"Experiment {exp_id} timed out")
            return experiment
            
        except Exception as e:
            experiment['status'] = 'error'
            experiment['completed_at'] = datetime.now().isoformat()
            experiment['error'] = str(e)
            
            self.logger.error(f"Experiment {exp_id} failed with error: {e}")
            return experiment
    
    def _parse_experiment_metrics(self, output_dir: Path) -> Dict[str, Any]:
        """Parse metrics from experiment output."""
        try:
            metrics = {}
            
            # Try to load experiment summary
            summary_file = output_dir / 'experiment_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    metrics.update(summary)
            
            # Try to load final metrics from trainer
            metrics_file = output_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    trainer_metrics = json.load(f)
                    metrics.update(trainer_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to parse metrics: {e}")
            return {}
    
    def _save_tracking_state(self):
        """Save current tracking state."""
        try:
            state = {
                'completed': self.completed_experiments,
                'failed': self.failed_experiments,
                'running': list(self.running_experiments.keys()),
                'total': len(self.experiments),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.tracking_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save tracking state: {e}")
    
    def _cleanup_running_experiments(self):
        """Clean up running experiments on shutdown."""
        try:
            for exp_id, process in self.running_experiments.items():
                if process.poll() is None:  # Still running
                    self.logger.info(f"Terminating experiment: {exp_id}")
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        
        except Exception as e:
            self.logger.error(f"Failed to cleanup experiments: {e}")
    
    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report."""
        try:
            self.logger.info("Generating benchmark report...")
            
            # Collect all results
            all_experiments = self.completed_experiments + self.failed_experiments
            
            # Generate summary statistics
            summary = {
                'total_experiments': len(all_experiments),
                'completed': len(self.completed_experiments),
                'failed': len(self.failed_experiments),
                'success_rate': len(self.completed_experiments) / len(all_experiments) if all_experiments else 0,
                'total_duration': sum(exp.get('duration', 0) for exp in self.completed_experiments),
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat()
            }
            
            # Algorithm performance comparison
            algorithm_results = {}
            for exp in self.completed_experiments:
                algo = exp['algorithm']
                if algo not in algorithm_results:
                    algorithm_results[algo] = []
                algorithm_results[algo].append(exp)
            
            # Generate performance metrics
            performance_metrics = {}
            for algo, exps in algorithm_results.items():
                if exps:
                    rewards = [exp.get('metrics', {}).get('reward_mean', 0) for exp in exps]
                    durations = [exp.get('duration', 0) for exp in exps]
                    
                    performance_metrics[algo] = {
                        'avg_reward': sum(rewards) / len(rewards),
                        'max_reward': max(rewards),
                        'avg_duration': sum(durations) / len(durations),
                        'total_duration': sum(durations),
                        'experiment_count': len(exps)
                    }
            
            # Create report
            report = {
                'summary': summary,
                'performance_metrics': performance_metrics,
                'experiments': all_experiments,
                'algorithm_comparison': algorithm_results
            }
            
            # Save report
            report_file = self.output_dir / 'benchmark_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate HTML report
            self._generate_html_report(report)
            
            self.logger.info("Benchmark report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to generate benchmark report: {e}")
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """Generate HTML benchmark report."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>RLHF Arena Benchmark Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .summary {{ margin: 20px 0; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                    .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                    .experiment-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    .experiment-table th, .experiment-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    .experiment-table th {{ background-color: #f2f2f2; }}
                    .status-completed {{ color: green; }}
                    .status-failed {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>RLHF Arena Benchmark Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p><strong>Total Experiments:</strong> {report['summary']['total_experiments']}</p>
                    <p><strong>Completed:</strong> {report['summary']['completed']}</p>
                    <p><strong>Failed:</strong> {report['summary']['failed']}</p>
                    <p><strong>Success Rate:</strong> {report['summary']['success_rate']:.2%}</p>
                    <p><strong>Total Duration:</strong> {report['summary']['total_duration']/3600:.2f} hours</p>
                </div>
                
                <div class="metrics">
                    <h2>Algorithm Performance</h2>
            """
            
            for algo, metrics in report['performance_metrics'].items():
                html_content += f"""
                    <div class="metric-card">
                        <h3>{algo.upper()}</h3>
                        <p><strong>Average Reward:</strong> {metrics['avg_reward']:.4f}</p>
                        <p><strong>Max Reward:</strong> {metrics['max_reward']:.4f}</p>
                        <p><strong>Average Duration:</strong> {metrics['avg_duration']/3600:.2f} hours</p>
                        <p><strong>Experiments:</strong> {metrics['experiment_count']}</p>
                    </div>
                """
            
            html_content += """
                </div>
                
                <h2>Experiment Details</h2>
                <table class="experiment-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Algorithm</th>
                            <th>Dataset</th>
                            <th>Status</th>
                            <th>Duration</th>
                            <th>Reward</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for exp in report['experiments']:
                status_class = f"status-{exp['status']}"
                duration = exp.get('duration', 0)
                reward = exp.get('metrics', {}).get('reward_mean', 'N/A')
                
                html_content += f"""
                    <tr>
                        <td>{exp['id']}</td>
                        <td>{exp['algorithm']}</td>
                        <td>{exp['dataset']}</td>
                        <td class="{status_class}">{exp['status']}</td>
                        <td>{duration/3600:.2f}h</td>
                        <td>{reward}</td>
                    </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            </body>
            </html>
            """
            
            # Save HTML report
            html_file = self.output_dir / 'benchmark_report.html'
            with open(html_file, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
    
    def run(self):
        """Run the complete benchmark."""
        try:
            self.start_time = datetime.now()
            self.logger.info("Starting benchmark...")
            
            # Generate experiments
            self._generate_experiments()
            
            # Run experiments with parallel execution
            with ProcessPoolExecutor(max_workers=self.max_parallel) as executor:
                # Submit all experiments
                future_to_exp = {
                    executor.submit(self._run_experiment, exp): exp 
                    for exp in self.experiments
                }
                
                # Process completed experiments
                for future in as_completed(future_to_exp):
                    experiment = future_to_exp[future]
                    try:
                        result = future.result()
                        
                        if result['status'] == 'completed':
                            self.completed_experiments.append(result)
                        else:
                            self.failed_experiments.append(result)
                        
                        # Update tracking
                        self._save_tracking_state()
                        
                        # Log progress
                        total = len(self.experiments)
                        completed = len(self.completed_experiments)
                        failed = len(self.failed_experiments)
                        self.logger.info(f"Progress: {completed + failed}/{total} experiments processed")
                        
                    except Exception as e:
                        self.logger.error(f"Experiment {experiment['id']} failed: {e}")
                        experiment['status'] = 'error'
                        experiment['error'] = str(e)
                        self.failed_experiments.append(experiment)
                        self._save_tracking_state()
            
            # Generate final report
            self._generate_benchmark_report()
            
            self.logger.info("Benchmark completed successfully!")
            self.logger.info(f"Completed: {len(self.completed_experiments)}, Failed: {len(self.failed_experiments)}")
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            self.logger.error(traceback.format_exc())
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RLHF Arena Benchmark Orchestrator")
    parser.add_argument("--config", required=True, help="Path to benchmark configuration file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_parallel", type=int, default=1, help="Maximum parallel experiments")
    
    args = parser.parse_args()
    
    try:
        # Create and run benchmark
        orchestrator = BenchmarkOrchestrator(
            config_path=args.config,
            output_dir=args.output_dir,
            max_parallel=args.max_parallel
        )
        
        orchestrator.run()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
