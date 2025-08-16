#!/usr/bin/env python3
"""
RLHF Arena Results Evaluator

Comprehensive evaluation script for analyzing experiment results,
generating comparison charts, and computing performance metrics.
"""

import argparse
import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rlhf_arena import load_config, setup_logging


class ResultsEvaluator:
    """Comprehensive results evaluator for RLHF experiments."""
    
    def __init__(self, results_dir: str, output_dir: str):
        """Initialize results evaluator."""
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.logger = None
        self.results = {}
        self.metrics_df = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load results
        self._load_results()
        
        # Process metrics
        self._process_metrics()
        
    def _setup_logging(self):
        """Setup logging."""
        log_config = {
            'output_dir': str(self.output_dir),
            'log_level': 'INFO'
        }
        self.logger = setup_logging(log_config)
        
        # Also log to file
        log_file = self.output_dir / 'evaluation.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Results evaluator initialized")
        self.logger.info(f"Results directory: {self.results_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _load_results(self):
        """Load all experiment results."""
        try:
            self.logger.info("Loading experiment results...")
            
            # Look for experiment directories
            experiment_dirs = []
            for item in self.results_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    experiment_dirs.append(item)
            
            self.logger.info(f"Found {len(experiment_dirs)} experiment directories")
            
            # Load results from each directory
            for exp_dir in experiment_dirs:
                exp_name = exp_dir.name
                self.logger.info(f"Loading results from: {exp_name}")
                
                # Try to load experiment summary
                summary_file = exp_dir / 'experiment_summary.json'
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                        self.results[exp_name] = summary
                        self.logger.info(f"Loaded summary for {exp_name}")
                
                # Try to load metrics
                metrics_file = exp_dir / 'metrics.json'
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        if exp_name in self.results:
                            self.results[exp_name].update(metrics)
                        else:
                            self.results[exp_name] = metrics
                        self.logger.info(f"Loaded metrics for {exp_name}")
                
                # Try to load config
                config_file = exp_dir / 'config.yaml'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if exp_name in self.results:
                            self.results[exp_name]['config'] = config
                        else:
                            self.results[exp_name] = {'config': config}
                        self.logger.info(f"Loaded config for {exp_name}")
            
            self.logger.info(f"Loaded results for {len(self.results)} experiments")
            
        except Exception as e:
            self.logger.error(f"Failed to load results: {e}")
            raise
    
    def _process_metrics(self):
        """Process and organize metrics into a DataFrame."""
        try:
            self.logger.info("Processing metrics...")
            
            # Extract key metrics from results
            metrics_data = []
            
            for exp_name, result in self.results.items():
                # Parse experiment name to extract algorithm, dataset, model_size
                parts = exp_name.split('_')
                if len(parts) >= 3:
                    algorithm = parts[0]
                    dataset = parts[1]
                    model_size = parts[2]
                else:
                    algorithm = result.get('config', {}).get('algorithm', 'unknown')
                    dataset = result.get('config', {}).get('dataset', {}).get('name', 'unknown')
                    model_size = 'unknown'
                
                # Extract metrics
                metrics = {
                    'experiment_name': exp_name,
                    'algorithm': algorithm,
                    'dataset': dataset,
                    'model_size': model_size,
                    'duration_hours': result.get('experiment_duration_hours', 0),
                    'reward_mean': result.get('final_metrics', {}).get('reward_mean', 0),
                    'reward_std': result.get('final_metrics', {}).get('reward_std', 0),
                    'kl_divergence': result.get('final_metrics', {}).get('kl_divergence', 0),
                    'entropy': result.get('final_metrics', {}).get('entropy', 0),
                    'policy_loss': result.get('final_metrics', {}).get('policy_loss', 0),
                    'value_loss': result.get('final_metrics', {}).get('value_loss', 0),
                    'clip_fraction': result.get('final_metrics', {}).get('clip_fraction', 0),
                    'best_reward': result.get('best_reward', 0),
                    'total_steps': result.get('final_metrics', {}).get('total_steps', 0),
                    'sample_efficiency': result.get('final_metrics', {}).get('sample_efficiency', 0),
                    'memory_usage_gb': result.get('final_metrics', {}).get('memory_usage_gb', 0),
                    'gpu_efficiency': result.get('final_metrics', {}).get('gpu_efficiency', 0)
                }
                
                metrics_data.append(metrics)
            
            # Create DataFrame
            self.metrics_df = pd.DataFrame(metrics_data)
            
            # Clean and validate data
            self.metrics_df = self.metrics_df.replace([np.inf, -np.inf], np.nan)
            self.metrics_df = self.metrics_df.dropna(subset=['reward_mean'])
            
            self.logger.info(f"Processed metrics for {len(self.metrics_df)} experiments")
            
            # Save processed metrics
            metrics_file = self.output_dir / 'processed_metrics.csv'
            self.metrics_df.to_csv(metrics_file, index=False)
            self.logger.info(f"Saved processed metrics to: {metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to process metrics: {e}")
            raise
    
    def generate_performance_comparison(self):
        """Generate algorithm performance comparison charts."""
        try:
            self.logger.info("Generating performance comparison charts...")
            
            if self.metrics_df.empty:
                self.logger.warning("No metrics data available for comparison")
                return
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('RLHF Algorithm Performance Comparison', fontsize=16, fontweight='bold')
            
            # 1. Average Reward by Algorithm
            ax1 = axes[0, 0]
            reward_data = self.metrics_df.groupby('algorithm')['reward_mean'].agg(['mean', 'std']).reset_index()
            bars = ax1.bar(reward_data['algorithm'], reward_data['mean'], 
                          yerr=reward_data['std'], capsize=5, alpha=0.8)
            ax1.set_title('Average Reward by Algorithm')
            ax1.set_ylabel('Reward')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
            
            # 2. Training Duration by Algorithm
            ax2 = axes[0, 1]
            duration_data = self.metrics_df.groupby('algorithm')['duration_hours'].mean()
            ax2.bar(duration_data.index, duration_data.values, alpha=0.8)
            ax2.set_title('Average Training Duration by Algorithm')
            ax2.set_ylabel('Duration (hours)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Sample Efficiency by Algorithm
            ax3 = axes[0, 2]
            efficiency_data = self.metrics_df.groupby('algorithm')['sample_efficiency'].mean()
            ax3.bar(efficiency_data.index, efficiency_data.values, alpha=0.8)
            ax3.set_title('Sample Efficiency by Algorithm')
            ax3.set_ylabel('Efficiency (reward/token)')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. KL Divergence by Algorithm
            ax4 = axes[1, 0]
            kl_data = self.metrics_df.groupby('algorithm')['kl_divergence'].mean()
            ax4.bar(kl_data.index, kl_data.values, alpha=0.8)
            ax4.set_title('KL Divergence by Algorithm')
            ax4.set_ylabel('KL Divergence')
            ax4.tick_params(axis='x', rotation=45)
            
            # 5. Memory Usage by Algorithm
            ax5 = axes[1, 1]
            memory_data = self.metrics_df.groupby('algorithm')['memory_usage_gb'].mean()
            ax5.bar(memory_data.index, memory_data.values, alpha=0.8)
            ax5.set_title('Memory Usage by Algorithm')
            ax5.set_ylabel('Memory (GB)')
            ax5.tick_params(axis='x', rotation=45)
            
            # 6. GPU Efficiency by Algorithm
            ax6 = axes[1, 2]
            gpu_data = self.metrics_df.groupby('algorithm')['gpu_efficiency'].mean()
            ax6.bar(gpu_data.index, gpu_data.values, alpha=0.8)
            ax6.set_title('GPU Efficiency by Algorithm')
            ax6.set_ylabel('GPU Efficiency (%)')
            ax6.tick_params(axis='x', rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / 'performance_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved performance comparison plot to: {plot_file}")
            
            # Also save as PDF
            pdf_file = self.output_dir / 'performance_comparison.pdf'
            plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance comparison: {e}")
    
    def generate_learning_curves(self):
        """Generate learning curves for each algorithm."""
        try:
            self.logger.info("Generating learning curves...")
            
            # Group by algorithm
            algorithms = self.metrics_df['algorithm'].unique()
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Learning Curves by Algorithm', fontsize=16, fontweight='bold')
            
            axes = axes.flatten()
            
            for i, algorithm in enumerate(algorithms):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Get data for this algorithm
                algo_data = self.metrics_df[self.metrics_df['algorithm'] == algorithm]
                
                # Plot reward vs duration
                ax.scatter(algo_data['duration_hours'], algo_data['reward_mean'], 
                          alpha=0.7, s=100, label=f'{algorithm.upper()}')
                
                # Add trend line
                if len(algo_data) > 1:
                    z = np.polyfit(algo_data['duration_hours'], algo_data['reward_mean'], 1)
                    p = np.poly1d(z)
                    ax.plot(algo_data['duration_hours'], p(algo_data['duration_hours']), 
                           "r--", alpha=0.8)
                
                ax.set_title(f'{algorithm.upper()} Learning Curve')
                ax.set_xlabel('Training Duration (hours)')
                ax.set_ylabel('Final Reward')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(algorithms), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / 'learning_curves.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved learning curves plot to: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate learning curves: {e}")
    
    def generate_dataset_comparison(self):
        """Generate dataset performance comparison."""
        try:
            self.logger.info("Generating dataset comparison...")
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Performance Across Datasets', fontsize=16, fontweight='bold')
            
            # 1. Average Reward by Dataset
            ax1 = axes[0, 0]
            dataset_reward = self.metrics_df.groupby('dataset')['reward_mean'].agg(['mean', 'std']).reset_index()
            bars = ax1.bar(dataset_reward['dataset'], dataset_reward['mean'], 
                          yerr=dataset_reward['std'], capsize=5, alpha=0.8)
            ax1.set_title('Average Reward by Dataset')
            ax1.set_ylabel('Reward')
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Training Duration by Dataset
            ax2 = axes[0, 1]
            dataset_duration = self.metrics_df.groupby('dataset')['duration_hours'].mean()
            ax2.bar(dataset_duration.index, dataset_duration.values, alpha=0.8)
            ax2.set_title('Average Training Duration by Dataset')
            ax2.set_ylabel('Duration (hours)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Algorithm Performance Heatmap
            ax3 = axes[1, 0]
            pivot_data = self.metrics_df.pivot_table(
                values='reward_mean', 
                index='algorithm', 
                columns='dataset', 
                aggfunc='mean'
            )
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax3)
            ax3.set_title('Algorithm-Dataset Performance Heatmap')
            
            # 4. Sample Efficiency by Dataset
            ax4 = axes[1, 1]
            dataset_efficiency = self.metrics_df.groupby('dataset')['sample_efficiency'].mean()
            ax4.bar(dataset_efficiency.index, dataset_efficiency.values, alpha=0.8)
            ax4.set_title('Sample Efficiency by Dataset')
            ax4.set_ylabel('Efficiency (reward/token)')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / 'dataset_comparison.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved dataset comparison plot to: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to generate dataset comparison: {e}")
    
    def generate_interactive_charts(self):
        """Generate interactive Plotly charts."""
        try:
            self.logger.info("Generating interactive charts...")
            
            # 1. 3D Scatter Plot: Algorithm vs Dataset vs Reward
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=self.metrics_df['algorithm'],
                y=self.metrics_df['dataset'],
                z=self.metrics_df['reward_mean'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.metrics_df['duration_hours'],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                text=self.metrics_df['experiment_name'],
                hovertemplate='<b>%{text}</b><br>' +
                            'Algorithm: %{x}<br>' +
                            'Dataset: %{y}<br>' +
                            'Reward: %{z:.3f}<br>' +
                            'Duration: %{marker.color:.2f}h<br>' +
                            '<extra></extra>'
            )])
            
            fig_3d.update_layout(
                title='3D Performance Visualization: Algorithm vs Dataset vs Reward',
                scene=dict(
                    xaxis_title='Algorithm',
                    yaxis_title='Dataset',
                    zaxis_title='Reward'
                ),
                width=1000,
                height=800
            )
            
            # Save 3D plot
            plot_3d_file = self.output_dir / '3d_performance.html'
            fig_3d.write_html(str(plot_3d_file))
            self.logger.info(f"Saved 3D performance plot to: {plot_3d_file}")
            
            # 2. Parallel Coordinates Plot
            if len(self.metrics_df) > 1:
                # Select numerical columns for parallel coordinates
                numerical_cols = ['reward_mean', 'duration_hours', 'kl_divergence', 
                                'sample_efficiency', 'memory_usage_gb']
                
                # Filter columns that exist in the data
                available_cols = [col for col in numerical_cols if col in self.metrics_df.columns]
                
                if available_cols:
                    fig_parallel = px.parallel_coordinates(
                        self.metrics_df[['algorithm'] + available_cols],
                        color='algorithm',
                        title='Parallel Coordinates: Algorithm Performance Comparison'
                    )
                    
                    parallel_file = self.output_dir / 'parallel_coordinates.html'
                    fig_parallel.write_html(str(parallel_file))
                    self.logger.info(f"Saved parallel coordinates plot to: {parallel_file}")
            
            # 3. Interactive Scatter Matrix
            if len(self.metrics_df) > 1:
                scatter_cols = ['reward_mean', 'duration_hours', 'kl_divergence']
                available_scatter_cols = [col for col in scatter_cols if col in self.metrics_df.columns]
                
                if len(available_scatter_cols) >= 2:
                    fig_scatter = px.scatter_matrix(
                        self.metrics_df[available_scatter_cols + ['algorithm']],
                        color='algorithm',
                        title='Scatter Matrix: Key Metrics Correlation',
                        dimensions=available_scatter_cols
                    )
                    
                    scatter_file = self.output_dir / 'scatter_matrix.html'
                    fig_scatter.write_html(str(scatter_file))
                    self.logger.info(f"Saved scatter matrix plot to: {scatter_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate interactive charts: {e}")
    
    def generate_statistical_summary(self):
        """Generate statistical summary of results."""
        try:
            self.logger.info("Generating statistical summary...")
            
            if self.metrics_df.empty:
                self.logger.warning("No metrics data available for statistical summary")
                return
            
            # Generate comprehensive statistics
            summary_stats = {}
            
            # Overall statistics
            summary_stats['overall'] = {
                'total_experiments': len(self.metrics_df),
                'algorithms_tested': self.metrics_df['algorithm'].nunique(),
                'datasets_tested': self.metrics_df['dataset'].nunique(),
                'total_training_time': self.metrics_df['duration_hours'].sum(),
                'average_reward': self.metrics_df['reward_mean'].mean(),
                'best_reward': self.metrics_df['reward_mean'].max(),
                'worst_reward': self.metrics_df['reward_mean'].min()
            }
            
            # Algorithm-specific statistics
            for algorithm in self.metrics_df['algorithm'].unique():
                algo_data = self.metrics_df[self.metrics_df['algorithm'] == algorithm]
                summary_stats[f'algorithm_{algorithm}'] = {
                    'experiment_count': len(algo_data),
                    'average_reward': algo_data['reward_mean'].mean(),
                    'reward_std': algo_data['reward_mean'].std(),
                    'best_reward': algo_data['reward_mean'].max(),
                    'average_duration': algo_data['duration_hours'].mean(),
                    'total_duration': algo_data['duration_hours'].sum(),
                    'sample_efficiency': algo_data['sample_efficiency'].mean() if 'sample_efficiency' in algo_data.columns else None,
                    'memory_usage': algo_data['memory_usage_gb'].mean() if 'memory_usage_gb' in algo_data.columns else None
                }
            
            # Dataset-specific statistics
            for dataset in self.metrics_df['dataset'].unique():
                dataset_data = self.metrics_df[self.metrics_df['dataset'] == dataset]
                summary_stats[f'dataset_{dataset}'] = {
                    'experiment_count': len(dataset_data),
                    'average_reward': dataset_data['reward_mean'].mean(),
                    'reward_std': dataset_data['reward_mean'].std(),
                    'best_reward': dataset_data['reward_mean'].max(),
                    'average_duration': dataset_data['duration_hours'].mean(),
                    'algorithms_tested': dataset_data['algorithm'].nunique()
                }
            
            # Save summary statistics
            summary_file = self.output_dir / 'statistical_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            self.logger.info(f"Saved statistical summary to: {summary_file}")
            
            # Generate summary report
            self._generate_summary_report(summary_stats)
            
        except Exception as e:
            self.logger.error(f"Failed to generate statistical summary: {e}")
    
    def _generate_summary_report(self, summary_stats: Dict[str, Any]):
        """Generate human-readable summary report."""
        try:
            report_content = f"""
# RLHF Arena Evaluation Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Summary

- **Total Experiments**: {summary_stats['overall']['total_experiments']}
- **Algorithms Tested**: {summary_stats['overall']['algorithms_tested']}
- **Datasets Tested**: {summary_stats['overall']['datasets_tested']}
- **Total Training Time**: {summary_stats['overall']['total_training_time']:.2f} hours
- **Average Reward**: {summary_stats['overall']['average_reward']:.4f}
- **Best Reward**: {summary_stats['overall']['best_reward']:.4f}
- **Worst Reward**: {summary_stats['overall']['worst_reward']:.4f}

## Algorithm Performance

"""
            
            # Add algorithm performance
            for key, value in summary_stats.items():
                if key.startswith('algorithm_'):
                    algo_name = key.replace('algorithm_', '').upper()
                    report_content += f"""
### {algo_name}
- **Experiments**: {value['experiment_count']}
- **Average Reward**: {value['average_reward']:.4f} ± {value['reward_std']:.4f}
- **Best Reward**: {value['best_reward']:.4f}
- **Average Duration**: {value['average_duration']:.2f} hours
- **Total Duration**: {value['total_duration']:.2f} hours
"""
                    if value['sample_efficiency'] is not None:
                        report_content += f"- **Sample Efficiency**: {value['sample_efficiency']:.4f}\n"
                    if value['memory_usage'] is not None:
                        report_content += f"- **Memory Usage**: {value['memory_usage']:.2f} GB\n"
            
            # Add dataset performance
            report_content += "\n## Dataset Performance\n"
            for key, value in summary_stats.items():
                if key.startswith('dataset_'):
                    dataset_name = key.replace('dataset_', '').upper()
                    report_content += f"""
### {dataset_name}
- **Experiments**: {value['experiment_count']}
- **Average Reward**: {value['average_reward']:.4f} ± {value['reward_std']:.4f}
- **Best Reward**: {value['best_reward']:.4f}
- **Average Duration**: {value['average_duration']:.2f} hours
- **Algorithms Tested**: {value['algorithms_tested']}
"""
            
            # Save report
            report_file = self.output_dir / 'evaluation_report.md'
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Saved evaluation report to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        try:
            self.logger.info("Starting complete evaluation pipeline...")
            
            # Generate all charts and reports
            self.generate_performance_comparison()
            self.generate_learning_curves()
            self.generate_dataset_comparison()
            self.generate_interactive_charts()
            self.generate_statistical_summary()
            
            self.logger.info("Evaluation pipeline completed successfully!")
            
            # Log summary
            if not self.metrics_df.empty:
                self.logger.info(f"Evaluated {len(self.metrics_df)} experiments")
                self.logger.info(f"Algorithms: {', '.join(self.metrics_df['algorithm'].unique())}")
                self.logger.info(f"Datasets: {', '.join(self.metrics_df['dataset'].unique())}")
                self.logger.info(f"Best reward: {self.metrics_df['reward_mean'].max():.4f}")
                self.logger.info(f"Total training time: {self.metrics_df['duration_hours'].sum():.2f} hours")
            
        except Exception as e:
            self.logger.error(f"Evaluation pipeline failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RLHF Arena Results Evaluator")
    parser.add_argument("--results_dir", required=True, help="Directory containing experiment results")
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation results")
    parser.add_argument("--compare", nargs='+', help="Specific algorithms to compare")
    parser.add_argument("--metrics", nargs='+', help="Specific metrics to analyze")
    
    args = parser.parse_args()
    
    try:
        # Create and run evaluator
        evaluator = ResultsEvaluator(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        
        # Run evaluation
        evaluator.run_evaluation()
        
        print(f"Evaluation completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
