# Neuro-Symbolic Code Generation AI - Replit Project

## Project Overview

This is a **production-grade neuro-symbolic code generation AI system** built with Python. The project combines neural networks with symbolic reasoning to achieve advanced code generation capabilities.

## Architecture

The system is designed with the following key components:

- **AST Encoder**: Language-agnostic code parsing and semantic embedding
- **SMT Connector**: Symbolic reasoning with Z3 for logical correctness
- **Neuro-Symbolic Orchestrator**: Adaptive pathway selection between neural and symbolic reasoning
- **Multi-dimensional Reward Model**: RLHF with test pass rate, security, quality, and license compliance
- **RAG Subsystem**: Hybrid retrieval with FAISS and Neo4j
- **Self-Repair Mechanism**: Automated defect detection and correction
- **Security Analysis Engine**: Comprehensive vulnerability scanning

## Quick Start

### 1. Run API Server

The API server is already running:

```bash
# API Server is running on http://0.0.0.0:5000
# Access it via:
# - Root: http://localhost:5000/
# - Health: http://localhost:5000/health
# - Swagger Docs: http://localhost:5000/docs
# - Generate Code: POST http://localhost:5000/generate
```

Example API usage:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "fibonacci function", "language": "python"}'
```

### 2. Full Setup (For Training)

Install Taskfile and run complete setup:

```bash
# Install Taskfile
brew install go-task  # macOS
# or use: sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# Run complete setup
task setup

# Download models
task download-models

# Prepare data
task prepare-data
```

### 3. Training

```bash
# Local training
task train

# Distributed training (4 GPUs)
task train:distributed

# Fine-tuning with LoRA
task finetune
```

### 4. Evaluation

```bash
# Run benchmarks
task evaluate

# Run tests
task test
```

## Project Structure

```
.
├── api_server.py                  # FastAPI production server (ACTIVE)
├── run_api.sh                     # API server launch script
├── src/neurosymbolic_codegen/     # Core modules
│   ├── core/                      # AST, SMT, Reward, Self-Repair
│   ├── orchestration/             # Neuro-Symbolic Orchestrator
│   ├── rag/                       # RAG Subsystem
│   ├── security/                  # Security Analysis
│   ├── api/                       # API modules (original)
│   ├── train.py                   # Training pipeline
│   └── demo.py                    # Interactive demo
├── configs/                       # Configuration files
├── tests/                         # Test suite
├── .argo-workflows/               # ArgoWorkflows manifests
├── docs/                          # Documentation
├── pyproject.toml                 # Dependencies (Poetry)
└── Taskfile.yml                   # Task automation
```

## Technology Stack

- **Core ML**: PyTorch, Transformers, DeepSpeed
- **Symbolic AI**: Z3, PySMT, tree-sitter
- **RAG**: FAISS, Neo4j, LangChain
- **Orchestration**: ArgoWorkflows, MLflow
- **Security**: Bandit, Semgrep, SPDX

## Cloud-Agnostic Design

This project is designed to run on any infrastructure:

- **Local**: Single machine or workstation
- **On-Premise**: Self-hosted GPU clusters
- **Cloud**: AWS, GCP, Azure (via ArgoWorkflows)
- **Hybrid**: Mix of resources

## Key Features

✅ Neuro-symbolic hybrid architecture
✅ Multi-agent coordination
✅ Adaptive reinforcement learning
✅ 128K context window support
✅ Self-repair mechanisms
✅ Comprehensive security analysis
✅ Cloud-agnostic deployment

## Available Tasks

Run `task --list` to see all available commands:

- `task setup` - Complete environment setup
- `task download-models` - Download pre-trained models
- `task prepare-data` - Prepare training data
- `task train` - Start training
- `task evaluate` - Run evaluation
- `task demo` - Run interactive demo
- `task mlflow:start` - Start MLflow server
- `task argo:install` - Install ArgoWorkflows

## Documentation

- [README.md](README.md) - Comprehensive overview
- [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) - Setup instructions
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture details
- [.argo-workflows/README.md](.argo-workflows/README.md) - Deployment guide

## Development Notes

- **Python 3.11.13** installed and configured (verified)
- Uses **Poetry 1.5.6** for dependency management (pyproject.toml)
- **Taskfile** for automated setup and deployment
- **ArgoWorkflows** for cloud-agnostic orchestration (not Tekton)
- All heavy dependencies (models, data) are downloaded via scripts
- No vendor lock-in - pure open-source stack

## Current Status

- ✅ Core architecture implemented
- ✅ Demo working
- ✅ Configuration files ready
- ✅ ArgoWorkflows manifests created
- ✅ Python 3.11 environment configured
- ✅ FastAPI production server running on port 5000
- ✅ API endpoints working (/health, /generate, /docs)
- ✅ Template-based code generation MVP functional
- ⏳ Ready for model training
- ⏳ Ready for deployment

## Developer Feedback / Known Limitations

The following components need enhancement for production:

### SMT Parser
- **Current State**: فعلاً ساده است (فقط basic comparisons)
- **Needs**: برای production پیچیده‌تر نیاز است
- Translation: Currently simple (only basic comparisons), needs to be more complex for production

### Self-Repair
- **Current State**: Fix generation کار می‌کند ولی محدود است  
- **Needs**: نیاز به AST-based rewrites دارد
- Translation: Fix generation works but is limited, needs AST-based rewrites

### FAISS
- **Current State**: Training logic هست
- **Needs**: نیاز به regression tests دارد
- Translation: Training logic exists but needs regression tests

## Next Steps

1. Install Taskfile and run `task setup`
2. Download models with `task download-models`
3. Start training with `task train`
4. Deploy with ArgoWorkflows: `task argo:install`

## License

Apache License 2.0
