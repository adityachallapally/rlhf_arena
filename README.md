# RLHF Arena: Benchmarking Frontier Post-Training RL Methods for LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive benchmarking framework for Reinforcement Learning from Human Feedback (RLHF) algorithms, designed to evaluate and compare the latest post-training methods for Large Language Models.

## 🚀 Features

- **Multiple RLHF Algorithms**: PPO, DPO, GRPO, Off-policy GRPO, GRPOVI, and RLAIF
- **Production-Ready**: Robust error handling, logging, and checkpointing
- **Configurable**: YAML-based configuration for all experiments
- **Multi-Dataset Support**: Anthropic HH, OpenAssistant, UltraFeedback
- **Comprehensive Metrics**: KL-divergence, rewards, sample efficiency, memory usage
- **Experiment Orchestration**: Automated multi-experiment benchmarking
- **Visualization**: Learning curves, comparison charts, and analysis reports

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ disk space for datasets

## 🛠️ Installation

### Option 1: From Source
```bash
git clone https://github.com/your-org/rlhf_arena.git
cd rlhf_arena
pip install -r requirements.txt
```

### Option 2: Docker
```bash
docker build -t rlhf_arena .
docker run -it --gpus all -v $(pwd):/workspace rlhf_arena
```

## 🚀 Quick Start

### 1. Basic PPO Training
```bash
python scripts/run_experiment.py --config configs/ppo.yaml --dataset hh --model_size 7b
```

### 2. Run All Algorithms Benchmark
```bash
python scripts/benchmark.py --config configs/multiobjective.yaml --datasets hh,oasst,ultrafeedback
```

### 3. Evaluate Results
```bash
python scripts/evaluate.py --results_dir reports/ --output_dir reports/analysis/
```

## 📁 Project Structure

```
rlhf_arena/
├── configs/                 # Configuration files
│   ├── ppo.yaml           # PPO algorithm config
│   ├── dpo.yaml           # DPO algorithm config
│   ├── grpo.yaml          # GRPO algorithm config
│   └── multiobjective.yaml # Multi-objective experiments
├── scripts/                # Main execution scripts
│   ├── run_experiment.py  # Single experiment runner
│   ├── benchmark.py       # Multi-experiment orchestrator
│   └── evaluate.py        # Results analysis
├── rlhf_arena/            # Core algorithm implementations
│   ├── ppo.py            # PPO trainer
│   ├── dpo.py            # DPO trainer
│   ├── grpo.py           # GRPO trainer
│   ├── grpo_offpolicy.py # Off-policy GRPO
│   ├── grpo_vi.py        # GRPO with value iteration
│   ├── rlaif.py          # RLAIF trainer
│   └── utils.py          # Common utilities
├── reports/               # Experiment results and reports
├── tests/                 # Unit tests
└── requirements.txt       # Dependencies
```

## ⚙️ Configuration

All experiments are configured via YAML files. Key configuration sections:

### Model Configuration
```yaml
model:
  checkpoint: "microsoft/DialoGPT-medium"
  max_length: 512
  temperature: 1.0
  top_p: 0.9
```

### Training Configuration
```yaml
training:
  batch_size: 4
  learning_rate: 1e-5
  num_epochs: 10
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
```

### Hardware Configuration
```yaml
hardware:
  device: "auto"  # auto, cuda, cpu
  mixed_precision: "fp16"
  gradient_checkpointing: true
```

## 🔬 Supported Algorithms

### 1. PPO (Proximal Policy Optimization)
- **Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **Features**: Value function learning, GAE, KL penalty, reward normalization
- **Use Case**: Standard RLHF with policy gradient methods

### 2. DPO (Direct Preference Optimization)
- **Paper**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **Features**: Direct preference learning, no reward model needed
- **Use Case**: Preference-based learning with human feedback

### 3. GRPO (Generalized Relative Policy Optimization)
- **Paper**: [GRPO: Generalized Relative Policy Optimization](https://arxiv.org/abs/2308.03270)
- **Features**: Relative policy optimization, improved sample efficiency
- **Use Case**: High-efficiency policy learning

### 4. GRPO Off-Policy
- **Features**: Off-policy learning, experience replay
- **Use Case**: Better sample efficiency with historical data

### 5. GRPO Value Iteration
- **Features**: Value iteration integration, improved convergence
- **Use Case**: Complex reward landscapes

### 6. RLAIF (Reinforcement Learning from AI Feedback)
- **Features**: AI-generated feedback, scalable training
- **Use Case**: Large-scale preference learning

## 📊 Datasets

### Anthropic HH (Helpful & Harmless)
- **Size**: ~160K conversations
- **Format**: Human-AI conversations with preference labels
- **Use Case**: General helpfulness and harmlessness

### OpenAssistant
- **Size**: ~160K conversations
- **Format**: Multi-turn conversations with quality scores
- **Use Case**: Conversational AI training

### UltraFeedback
- **Size**: ~64K preference pairs
- **Format**: High-quality preference pairs
- **Use Case**: Preference learning research

## 📈 Metrics & Evaluation

### Training Metrics
- Policy loss, value loss, entropy
- KL divergence from reference model
- Reward statistics (mean, std)
- Clip fractions and learning rates

### Evaluation Metrics
- **Sample Efficiency**: Tokens to achieve reward threshold
- **Memory Usage**: GPU memory consumption
- **Convergence**: Learning curve analysis
- **Quality**: Human evaluation scores

### Visualization
- Learning curves and reward plots
- Algorithm comparison charts
- Memory usage over time
- Sample efficiency analysis

## 🧪 Running Experiments

### Single Algorithm Training
```bash
python scripts/run_experiment.py \
    --config configs/ppo.yaml \
    --dataset hh \
    --model_size 7b \
    --output_dir experiments/ppo_hh_7b
```

### Multi-Algorithm Benchmark
```bash
python scripts/benchmark.py \
    --config configs/multiobjective.yaml \
    --algorithms ppo,dpo,grpo \
    --datasets hh,oasst \
    --output_dir experiments/benchmark
```

### Custom Configuration
```bash
python scripts/run_experiment.py \
    --config configs/custom.yaml \
    --override training.batch_size=8 training.learning_rate=2e-5
```

## 📊 Results Analysis

### Generate Reports
```bash
python scripts/evaluate.py \
    --results_dir experiments/ \
    --output_dir reports/ \
    --format html,pdf
```

### Compare Algorithms
```bash
python scripts/evaluate.py \
    --compare ppo dpo grpo \
    --metrics reward_mean sample_efficiency memory_usage \
    --output_dir reports/comparison
```

## 🐳 Docker Usage

### Build Image
```bash
docker build -t rlhf_arena .
```

### Run Container
```bash
docker run -it --gpus all \
    -v $(pwd):/workspace \
    -v /path/to/datasets:/datasets \
    rlhf_arena
```

### Multi-GPU Training
```bash
docker run -it --gpus all \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -v $(pwd):/workspace \
    rlhf_arena
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_ppo.py -v
pytest tests/test_utils.py -v
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Hugging Face for transformers and datasets
- PyTorch team for the deep learning framework
- Anthropic, OpenAssistant, and UltraFeedback teams for datasets
- The RLHF research community

## 📚 References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [GRPO Paper](https://arxiv.org/abs/2308.03270)
- [RLAIF Paper](https://arxiv.org/abs/2309.00267)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/rlhf_arena/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rlhf_arena/discussions)
- **Email**: support@rlhf-arena.org

---

**Note**: This is a research tool. Results may vary based on hardware, datasets, and hyperparameters. Always validate results on your specific use case.
