# Setup Guide - Neuro-Symbolic Code Generation AI

This guide walks you through setting up the Neuro-Symbolic Code Generation AI system from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Download Models](#download-models)
5. [Prepare Data](#prepare-data)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, for training)
- **Memory**: 32GB RAM minimum (64GB+ recommended for training)
- **Storage**: 200GB+ free disk space

### Required Software

- **Git**: For cloning the repository
- **Taskfile**: For running automated scripts
- **Poetry**: Python dependency management (installed automatically)
- **Docker** (optional): For containerized deployment
- **Kubernetes** (optional): For ArgoWorkflows

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/neurosymbolic-codegen-ai.git
cd neurosymbolic-codegen-ai
```

### Step 2: Install Taskfile

Taskfile automates all setup and training tasks.

**On macOS:**
```bash
brew install go-task/tap/go-task
```

**On Linux:**
```bash
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
```

### Step 3: Run Complete Setup

This installs all dependencies, creates directories, and sets up the environment:

```bash
task setup
```

This command will:
- Check Python version
- Install Poetry
- Install Python dependencies
- Set up tree-sitter grammars
- Create necessary directories
- Configure MLflow

---

## Configuration

### Training Configuration

Edit `configs/training_config.yaml` to customize training parameters:

```yaml
training:
  batch_size: 16
  num_epochs: 3
  learning_rate: 2.0e-5
```

### Fine-tuning Configuration

Edit `configs/finetune_config.yaml` for LoRA/PEFT settings:

```yaml
peft:
  lora:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Evaluation Configuration

Edit `configs/eval_config.yaml` for benchmark settings:

```yaml
benchmarks:
  humaneval:
    enabled: true
    num_samples: 164
```

---

## Download Models

### Pre-trained Models

Download base models from HuggingFace:

```bash
task download-models
```

This provides interactive prompts to download:
- DeepSeek-Coder-6.7B
- StarCoder2-15B
- CodeLlama-13B

**Manual Download:**

```bash
huggingface-cli download deepseek-ai/deepseek-coder-6.7b-base \
  --local-dir models/pretrained/deepseek-coder-6.7b
```

---

## Prepare Data

### Download Training Data

```bash
task prepare-data
```

This downloads and processes code datasets from:
- The Stack (HuggingFace)
- CodeSearchNet
- GitHub Code Dataset

### Download Benchmark Datasets

```bash
task download-benchmarks
```

Downloads:
- HumanEval++
- MBPP+
- CodeContests

---

## Training

### Local Training

For single-GPU or CPU training:

```bash
task train
```

### Distributed Training

For multi-GPU training with DeepSpeed:

```bash
task train:distributed
```

This uses 4 GPUs by default. Edit `configs/training_config.yaml` to change.

### Fine-tuning with LoRA

Memory-efficient fine-tuning:

```bash
task finetune
```

Uses QLoRA (4-bit quantization) for efficient training on limited hardware.

### Monitor Training

Start MLflow tracking server:

```bash
task mlflow:start
```

Open MLflow UI:

```bash
task mlflow:ui
```

Navigate to http://localhost:5000 to view:
- Training metrics
- Loss curves
- Model versions
- Experiment comparisons

---

## Evaluation

### Run Benchmarks

Evaluate on all benchmarks:

```bash
task evaluate
```

### Run Specific Tests

Unit tests:

```bash
task test:unit
```

Integration tests:

```bash
task test:integration
```

Benchmark tests:

```bash
task test:benchmarks
```

### View Results

Results are saved to:
- `logs/evaluation/results.json`
- `logs/evaluation/report.md`

---

## Deployment

### Cloud-Agnostic Deployment with ArgoWorkflows

#### Prerequisites

1. Kubernetes cluster (any provider)
2. ArgoWorkflows installed

#### Install Workflows

```bash
task argo:install
```

#### Submit Training Workflow

```bash
task argo:submit:train
```

#### Submit Evaluation Workflow

```bash
task argo:submit:evaluate
```

#### Monitor Workflows

```bash
argo list
argo watch <workflow-name>
argo logs <workflow-name>
```

### Local Deployment

Run the demo:

```bash
task demo
```

---

## Troubleshooting

### Common Issues

#### 1. GPU Out of Memory

**Solution:** Reduce batch size or enable gradient checkpointing:

```yaml
# configs/training_config.yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 8

memory_optimization:
  gradient_checkpointing: true
```

#### 2. Dependency Installation Fails

**Solution:** Reinstall dependencies:

```bash
poetry cache clear pypi --all
task install-deps
```

#### 3. Tree-sitter Setup Fails

**Solution:** Manually install tree-sitter grammars:

```bash
task install-tree-sitter-languages
```

#### 4. MLflow Connection Error

**Solution:** Restart MLflow server:

```bash
pkill -f mlflow
task mlflow:start
```

#### 5. CUDA Not Available

**Solution:** Install CUDA 11.8+ or use CPU mode:

```yaml
# configs/training_config.yaml
training:
  device: "cpu"
```

---

## Next Steps

1. **Customize the model**: Edit model architecture in source code
2. **Add new benchmarks**: Create custom evaluation tasks
3. **Deploy to production**: Use ArgoWorkflows for scalable deployment
4. **Contribute**: Submit pull requests with improvements

---

## Support

For issues and questions:
- Open an issue on GitHub
- Check documentation in `docs/`
- Run `task help` for command reference

---

## References

- [Architecture Documentation](ARCHITECTURE.md)
- [Training Guide](TRAINING.md)
- [API Reference](API.md)
- [ArgoWorkflows Guide](../.argo-workflows/README.md)
