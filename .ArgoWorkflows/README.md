# ArgoWorkflows Manifests

This directory contains ArgoWorkflows manifests for cloud-agnostic distributed training, fine-tuning, and evaluation of the Neuro-Symbolic Code Generation AI.

## Prerequisites

1. **Kubernetes Cluster** (any provider: AWS EKS, GCP GKE, Azure AKS, self-hosted)
2. **ArgoWorkflows** installed on the cluster
3. **GPU nodes** for training (optional for CPU-only evaluation)

## Installation

### 1. Install ArgoWorkflows on your cluster

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml
```

### 2. Install Argo CLI

```bash
# macOS
brew install argo

# Linux
curl -sLO https://github.com/argoproj/argo-workflows/releases/latest/download/argo-linux-amd64.gz
gunzip argo-linux-amd64.gz
chmod +x argo-linux-amd64
sudo mv argo-linux-amd64 /usr/local/bin/argo
```

## Available Workflows

### 1. Training Workflow (`training-workflow.yaml`)

Full distributed training pipeline with DeepSpeed.

**Submit:**
```bash
argo submit .argo-workflows/training-workflow.yaml \
  --parameter model-name="deepseek-coder-6.7b" \
  --parameter batch-size="16" \
  --parameter num-epochs="3" \
  --parameter learning-rate="2e-5" \
  --parameter num-gpus="4"
```

**Steps:**
1. Environment setup
2. Download pre-trained model
3. Prepare training data
4. Distributed training with DeepSpeed
5. Model validation
6. Save artifacts

### 2. Fine-tuning Workflow (`finetune-workflow.yaml`)

Memory-efficient fine-tuning with LoRA/PEFT.

**Submit:**
```bash
argo submit .argo-workflows/finetune-workflow.yaml \
  --parameter base-model="deepseek-coder-6.7b" \
  --parameter use-lora="true" \
  --parameter lora-rank="16" \
  --parameter lora-alpha="32" \
  --parameter target-modules="q_proj,v_proj,k_proj,o_proj"
```

**Steps:**
1. Setup environment
2. Load base model
3. Configure PEFT (LoRA)
4. Run fine-tuning
5. Merge LoRA adapters (if applicable)
6. Save fine-tuned model

### 3. Evaluation Workflow (`evaluation-workflow.yaml`)

Comprehensive benchmark evaluation.

**Submit:**
```bash
argo submit .argo-workflows/evaluation-workflow.yaml \
  --parameter model-checkpoint="/workspace/models/finetuned/latest" \
  --parameter benchmarks="humaneval,mbpp,codecontests"
```

**Steps:**
1. Setup evaluation environment
2. Download benchmark datasets
3. Parallel evaluation on multiple benchmarks
4. Aggregate metrics
5. Generate evaluation report

## Monitoring Workflows

### List all workflows
```bash
argo list
```

### Watch workflow progress
```bash
argo watch <workflow-name>
```

### Get workflow logs
```bash
argo logs <workflow-name>
```

### Get workflow details
```bash
argo get <workflow-name>
```

## Workflow Parameters

### Training Workflow

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model-name` | Base model to train | `deepseek-coder-6.7b` |
| `batch-size` | Training batch size | `16` |
| `num-epochs` | Number of training epochs | `3` |
| `learning-rate` | Learning rate | `2e-5` |
| `num-gpus` | Number of GPUs for distributed training | `4` |

### Fine-tuning Workflow

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base-model` | Base model to fine-tune | `deepseek-coder-6.7b` |
| `use-lora` | Use LoRA for parameter-efficient fine-tuning | `true` |
| `lora-rank` | LoRA rank | `16` |
| `lora-alpha` | LoRA alpha | `32` |
| `target-modules` | Target modules for LoRA | `q_proj,v_proj,k_proj,o_proj` |

### Evaluation Workflow

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model-checkpoint` | Model checkpoint path | `/workspace/models/finetuned/latest` |
| `benchmarks` | Comma-separated benchmark names | `humaneval,mbpp,codecontests` |

## Cloud Provider Examples

### AWS EKS

```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name neurosymbolic-ai \
  --region us-west-2 \
  --nodegroup-name gpu-nodes \
  --node-type p3.8xlarge \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4

# Submit workflow
argo submit .argo-workflows/training-workflow.yaml
```

### GCP GKE

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create neurosymbolic-ai \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-v100,count=2 \
  --num-nodes 2

# Submit workflow
argo submit .argo-workflows/training-workflow.yaml
```

### Azure AKS

```bash
# Create AKS cluster with GPU nodes
az aks create \
  --resource-group neurosymbolic-rg \
  --name neurosymbolic-ai \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler

# Submit workflow
argo submit .argo-workflows/training-workflow.yaml
```

## Customization

### Add custom steps

Edit the workflow YAML and add new templates:

```yaml
- name: custom-step
  container:
    image: your-image:tag
    command: ["/bin/bash", "-c"]
    args:
      - |
        # Your custom logic here
```

### Use different container images

Replace the default images with your preferred ones:

```yaml
container:
  image: your-registry/your-image:tag
```

### Configure resource limits

Adjust CPU/GPU/memory limits:

```yaml
resources:
  limits:
    nvidia.com/gpu: "4"
    memory: "32Gi"
    cpu: "16"
  requests:
    memory: "16Gi"
    cpu: "8"
```

## Troubleshooting

### Workflow fails with ImagePullBackOff

Ensure your cluster has access to the container registry:

```bash
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password>
```

### Out of GPU memory

Reduce batch size or use gradient accumulation:

```bash
argo submit training-workflow.yaml --parameter batch-size="8"
```

### Permission errors

Check service account permissions:

```bash
kubectl create rolebinding argo-admin \
  --clusterrole=admin \
  --serviceaccount=argo:default
```

## Best Practices

1. **Use persistent volumes** for model and data storage
2. **Monitor GPU utilization** with prometheus/grafana
3. **Set resource limits** to prevent resource exhaustion
4. **Use workflow templates** for reusable patterns
5. **Enable artifact repository** for storing outputs
6. **Configure retry policies** for transient failures

## Integration with CI/CD

### GitHub Actions

```yaml
name: Train Model
on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Submit Argo Workflow
        run: |
          argo submit .argo-workflows/training-workflow.yaml \
            --wait --log
```

### GitLab CI

```yaml
train_model:
  script:
    - argo submit .argo-workflows/training-workflow.yaml --wait --log
  only:
    - main
```

## References

- [ArgoWorkflows Documentation](https://argoproj.github.io/argo-workflows/)
- [Workflow Examples](https://github.com/argoproj/argo-workflows/tree/master/examples)
- [Best Practices](https://argoproj.github.io/argo-workflows/workflow-concepts/)
