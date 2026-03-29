# Yadabo AI

A complete Transformer-based language model built from scratch, optimized for programming and mathematical tasks. Supports both Arabic and English languages.

## Features

- **Custom Transformer Architecture**: Built from scratch using PyTorch
- **BPE Tokenizer**: Custom Byte Pair Encoding tokenizer supporting Arabic and English
- **Modular Design**: Clean separation of model, training, data, and utilities
- **Production-Ready**: Checkpointing, logging, and scalability support
- **GPU-Optimized**: Ready for Google Cloud A100 and similar GPUs

## Quick Start

### Installation

```bash
pip install torch numpy tqdm
```

### Training

```bash
python scripts/train.py --config config/default.yaml
```

### Text Generation

```bash
python scripts/generate.py --checkpoint checkpoints/yadabo_best.pt --prompt "def hello_world():"
```

## Project Structure

```
yadabo-ai/
├── model/              # Transformer architecture
├── training/           # Training loop and optimizer
├── data/              # Dataset and tokenizer
├── utils/             # Helper functions
├── scripts/           # Training and inference scripts
├── config/            # Configuration files
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Architecture

The model implements a complete Transformer architecture:

- Token Embeddings with learned positional encodings
- Multi-head self-attention mechanism
- Feed-forward networks with GELU activation
- Layer normalization and dropout
- Custom BPE tokenizer for Arabic and English

## Configuration

Default configuration targets:
- Model: 6 layers, 8 attention heads, 512 hidden dimension
- Training: AdamW optimizer, cosine learning rate schedule
- Batch size: 32, Sequence length: 512

For production training on A100:
- Model: 12 layers, 16 heads, 1024 hidden dimension
- Batch size: 64, Sequence length: 2048

## License

MIT License
# Yadabo AI - API Reference

Complete API reference for Yadabo AI.

## Model

### TransformerLM

Main transformer language model class.

```python
from model import TransformerLM, TransformerConfig

# Create configuration
config = TransformerConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
)

# Create model
model = TransformerLM(config)

# Forward pass
input_ids = torch.randint(0, 32000, (2, 100))  # (batch, seq)
logits = model(input_ids)  # (batch, seq, vocab_size)
```

#### Methods

##### `forward(input_ids, attention_mask=None, labels=None)`

Forward pass through the model.

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs, shape (batch_size, seq_len)
- `attention_mask` (torch.Tensor, optional): Attention mask
- `labels` (torch.Tensor, optional): Labels for loss computation

**Returns:**
- If `labels` provided: Tuple of (loss, logits)
- Otherwise: Logits tensor

##### `generate(input_ids, **kwargs)`

Generate text autoregressively.

**Parameters:**
- `input_ids` (torch.Tensor): Starting token IDs
- `max_new_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `top_k` (int, optional): Top-k sampling
- `top_p` (float, optional): Nucleus sampling
- `repetition_penalty` (float): Repetition penalty

**Returns:**
- Generated token IDs

##### `beam_search(input_ids, max_new_tokens, beam_size=5)`

Beam search generation.

##### `save(path)` / `load(path)`

Save and load model checkpoints.

---

### TransformerConfig

Configuration dataclass for the model.

```python
config = TransformerConfig(
    vocab_size=32000,
    max_seq_len=512,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    activation="gelu",
)
```

**Presets:**

```python
TransformerConfig.small_arabic_english()   # ~8M params
TransformerConfig.medium_arabic_english()    # ~40M params
TransformerConfig.large_arabic_english()    # ~125M params
TransformerConfig.xlarge_arabic_english()   # ~350M params
```

---

## Data

### BPETokenizer

Custom BPE tokenizer for Arabic and English.

```python
from data import BPETokenizer

# Load trained tokenizer
tokenizer = BPETokenizer.load("tokenizer.json")

# Encode
input_ids = tokenizer.encode("Hello, world!")
tokens = tokenizer.tokenize("Hello, world!")

# Decode
text = tokenizer.decode(input_ids)
```

#### Methods

##### `encode(text, add_special_tokens=True, max_length=None)`

Encode text to token IDs.

##### `decode(token_ids, skip_special_tokens=True)`

Decode token IDs to text.

##### `train(corpus, vocab_size, min_frequency)`

Train tokenizer on corpus.

##### `batch_encode()` / `batch_decode()`

Batch operations.

---

### TextDataset

Dataset for loading text data.

```python
from data import TextDataset, DataCollator, create_dataloader

dataset = TextDataset(
    file_path="train.txt",
    tokenizer=tokenizer,
    max_length=512,
    stride=128,
)

collator = DataCollator(tokenizer=tokenizer)
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collator,
)
```

---

## Training

### Trainer

Main training class.

```python
from training import Trainer, TrainerConfig

config = TrainerConfig(
    output_dir="./output",
    max_steps=100000,
    learning_rate=1e-4,
    batch_size=32,
)

trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=val_loader,
)

trainer.train()
```

### Optimizer

```python
from training import create_optimizer

optimizer = create_optimizer(
    model,
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
)
```

**Supported optimizers:**
- `adamw` - Adam with weight decay
- `adam` - Standard Adam
- `lion` - Lion optimizer
- `sophia` - Sophia optimizer
- `lamb` - LAMB optimizer

### Scheduler

```python
from training import create_scheduler

scheduler = create_scheduler(
    optimizer,
    scheduler_type="cosine",
    num_warmup_steps=1000,
    num_training_steps=100000,
    min_lr=0.0,
)
```

**Supported schedulers:**
- `cosine` - Cosine annealing with warmup
- `linear` - Linear decay with warmup
- `polynomial` - Polynomial decay
- `constant` - Constant with warmup

---

## Utils

### Logging

```python
from utils import setup_logger

logger = setup_logger(
    name="yadabo",
    log_dir="./logs",
    level=logging.INFO,
)
```

### Device

```python
from utils import get_device, set_seed

device = get_device("auto")  # auto, cuda, cpu, mps
set_seed(42)
```

### Model Utils

```python
from utils import count_parameters, freeze_layers

total, trainable = count_parameters(model)
freeze_layers(model, freeze_embeddings=True)
```

### Checkpoint Utils

```python
from utils import save_checkpoint, load_checkpoint, get_latest_checkpoint

save_checkpoint(model, optimizer, step=1000, ...)
checkpoint = get_latest_checkpoint("./output")
```

---

## Scripts

### Training

```bash
python scripts/train.py \
    --config config/default.yaml \
    --data_path data/train.txt \
    --output_dir ./output
```

### Generation

```bash
python scripts/generate.py \
    --checkpoint output/best.pt \
    --prompt "def hello_world():" \
    --max_new_tokens 100
```

### Tokenizer Training

```bash
python scripts/train_tokenizer.py \
    --data data/train.txt \
    --output data/tokenizer.json \
    --vocab_size 32000
```
# Yadabo AI - Architecture Documentation

This document provides detailed information about Yadabo AI's architecture.

## Overview

Yadabo AI is a Transformer-based language model built from scratch using PyTorch. It implements the complete Transformer architecture with support for both Arabic and English languages.

## Architecture Components

### 1. Token Embedding Layer

The token embedding layer transforms token IDs into dense vectors.

```python
TokenEmbedding(
    vocab_size=32000,
    d_model=512,
    padding_idx=0,
    dropout=0.1
)
```

**Features:**
- Learned embeddings for each token
- Optional dropout for regularization
- Handles padding tokens specially

### 2. Positional Encoding

Two options are available:

#### Sinusoidal Positional Encoding
- Uses sine and cosine functions
- Allows model to attend to relative positions
- No learned parameters

```python
PositionalEncoding(
    d_model=512,
    max_len=2048,
    dropout=0.1
)
```

#### Learned Positional Encoding
- Learns position embeddings
- May perform better for some tasks
- More parameters

### 3. Transformer Block

Each transformer block contains:

1. **Multi-Head Self-Attention**
   - 8 attention heads (configurable)
   - Scaled dot-product attention
   - Causal masking for autoregressive generation

2. **Feed-Forward Network**
   - Two linear layers with GELU activation
   - 4x expansion ratio (d_model -> d_ff -> d_model)

3. **Residual Connections**
   - Pre-normalization architecture
   - Layer normalization before each sublayer

```python
TransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    activation="gelu"
)
```

### 4. Multi-Head Attention

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Variants:**
- Standard Multi-Head Attention
- Grouped Query Attention (GQA) - fewer K/V heads
- Cross Attention (for encoder-decoder)

### 5. Feed-Forward Network

Standard FFN with configurable activation:

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

**Activation options:**
- GELU (default)
- ReLU
- SwiGLU (LLaMA-style)

## Model Variants

### Small (Testing)
```yaml
d_model: 256
n_heads: 4
n_layers: 4
d_ff: 1024
Parameters: ~8M
GPU Memory: ~1GB
```

### Medium (Development)
```yaml
d_model: 512
n_heads: 8
n_layers: 6
d_ff: 2048
Parameters: ~40M
GPU Memory: ~2GB
```

### Large (Production)
```yaml
d_model: 768
n_heads: 12
n_layers: 12
d_ff: 3072
Parameters: ~125M
GPU Memory: ~6GB
```

### Extra Large (Maximum)
```yaml
d_model: 1024
n_heads: 16
n_layers: 24
d_ff: 4096
Parameters: ~350M
GPU Memory: ~16GB
```

## Training

### Optimization
- **Optimizer:** AdamW with weight decay
- **Learning Rate Schedule:** Cosine annealing with warmup
- **Gradient Clipping:** max_norm=1.0
- **Gradient Checkpointing:** Optional for memory efficiency

### Regularization
- Dropout (configurable)
- Weight decay
- Label smoothing (optional)

## Tokenizer

### BPE Tokenizer

Byte Pair Encoding tokenizer optimized for Arabic and English:

1. **Pre-tokenization:** Splits text into words
2. **Byte-level encoding:** Handles any Unicode character
3. **BPE merging:** Learns most common token pairs
4. **Special tokens:** <pad>, <bos>, <eos>, <unk>, <mask>

### Arabic Support

- Full Arabic Unicode range (U+0600 to U+06FF)
- Proper handling of Arabic diacritics
- RTL text support

## Inference

### Generation Methods

1. **Greedy Decoding:** Select most probable token
2. **Temperature Sampling:** Add randomness
3. **Top-K Sampling:** Sample from top K tokens
4. **Nucleus (Top-P) Sampling:** Dynamic token selection
5. **Beam Search:** Explore multiple hypotheses

### KV Cache

For efficient generation, the model caches key and value tensors from previous steps.

## Memory Optimization

### Techniques Used

1. **Gradient Checkpointing:** Trade compute for memory
2. **Mixed Precision:** FP16/BF16 training
3. **Gradient Accumulation:** Larger effective batch size
4. **Efficient Data Loading:** Prefetching and pinning

### Memory Estimates

| Configuration | Parameters | Batch Size | Memory |
|--------------|------------|------------|--------|
| Small | 8M | 32 | 2GB |
| Medium | 40M | 16 | 4GB |
| Large | 125M | 8 | 8GB |
| XLarge | 350M | 4 | 16GB |

## Extensibility

### Adding New Attention Types

```python
class CustomAttention(MultiHeadAttention):
    def forward(self, x):
        # Custom attention logic
        pass

# Register in model
TransformerBlock(attention=CustomAttention)
```

### Custom Tokenizer

```python
class CustomTokenizer(BPETokenizer):
    def _init_patterns(self):
        # Add custom tokenization patterns
        pass
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
# Yadabo AI - Quick Start Guide

Get started with Yadabo AI in minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/yadabo/yadabo-ai.git
cd yadabo-ai

# Install dependencies
pip install torch numpy tqdm

# Install Yadabo AI
pip install -e .
```

## Quick Training Example

### 1. Prepare Data

Create a simple text file:

```bash
echo "Hello, world!
This is Yadabo AI.
We are training a language model." > data/sample.txt
```

### 2. Train Tokenizer

```bash
python scripts/train_tokenizer.py \
    --data data/sample.txt \
    --output data/tokenizer.json \
    --vocab_size 5000
```

### 3. Train Model

```bash
python scripts/train.py \
    --config config/small_test.yaml \
    --data_path data/sample.txt \
    --tokenizer_path data/tokenizer.json \
    --max_steps 100 \
    --output_dir ./output
```

### 4. Generate Text

```bash
python scripts/generate.py \
    --checkpoint ./output/checkpoint_step_100.pt \
    --tokenizer data/tokenizer.json \
    --prompt "Hello" \
    --max_new_tokens 50
```

## Complete Example Script

```python
import torch
from model import TransformerLM, TransformerConfig
from data import BPETokenizer, TextDataset, DataCollator, create_dataloader
from training import Trainer, TrainerConfig

# 1. Create tokenizer
tokenizer = BPETokenizer()
tokenizer.train(
    corpus=["Your training text here..."] * 100,
    vocab_size=5000,
)

# 2. Create model
config = TransformerConfig.small_arabic_english()
config.vocab_size = 5000
model = TransformerLM(config)

# 3. Create dataset
dataset = TextDataset(
    file_path="data/sample.txt",
    tokenizer=tokenizer,
    max_length=128,
)

# 4. Create trainer
trainer = Trainer(
    model=model,
    config=TrainerConfig(
        max_steps=100,
        learning_rate=1e-3,
        batch_size=4,
    ),
    train_dataloader=create_dataloader(
        dataset=dataset,
        batch_size=4,
        collate_fn=DataCollator(tokenizer),
    ),
)

# 5. Train
trainer.train()

# 6. Generate
prompt = "Hello"
input_ids = tokenizer.encode(prompt)
output = model.generate(
    torch.tensor([input_ids]),
    max_new_tokens=50,
)
print(tokenizer.decode(output[0]))
```

## Next Steps

- Read the [full documentation](README.md)
- Check out [configuration options](config/)
- Learn about [Google Cloud setup](setup_gcp.md)
- Explore the [API reference](api_reference.md)

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint output/best.pt \
    --data data/test.txt \
    --tokenizer data/tokenizer.json
```
# Yadabo AI - Google Cloud Setup Guide

This guide explains how to set up Yadabo AI for training on Google Cloud Platform (GCP) with GPU support.

## Prerequisites

1. Google Cloud account with billing enabled
2. gcloud CLI installed and configured
3. Project with sufficient quota for GPU instances

## Quick Start with Cloud GPU

### 1. Create a GPU Instance

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create A100 instance
gcloud compute instances create yadabo-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --image-family=pytorch-latest \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --scopes=storage-full
```

### 2. Connect to Instance

```bash
gcloud compute ssh yadabo-training --zone=us-central1-a
```

### 3. Install Yadabo AI

```bash
# Clone repository
git clone https://github.com/yadabo/yadabo-ai.git
cd yadabo-ai

# Install dependencies
pip install -e .

# Verify PyTorch with CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Setup on GCP

### Option 1: Cloud Storage Bucket

```bash
# Create bucket
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME

# Upload training data
gsutil cp -r data/* gs://YOUR_BUCKET_NAME/data/

# Update config to use GCS paths
# Or use symbolic links
ln -sf /path/to/gcs/data ./data
```

### Option 2: Local Storage with Cloud Sync

```bash
# Sync data periodically
gsutil rsync -r gs://YOUR_BUCKET_NAME/data ./data
```

## Multi-GPU Training

### Create Multi-GPU Instance

```bash
# Create instance with 4 A100s
gcloud compute instances create yadabo-training-4x \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-4g \
    --accelerator=type=nvidia-tesla-a100,count=4 \
    --image-family=pytorch-latest \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=500GB \
    --boot-disk-type=pd-ssd
```

### Launch Distributed Training

```bash
# Using torchrun for distributed training
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=123 \
    --rdzv_endpoint="localhost:29500" \
    scripts/train.py \
    --config config/production.yaml \
    --output_dir gs://YOUR_BUCKET_NAME/output
```

## Scaling Training

### Horizontal Scaling (Multiple Nodes)

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --rdzv_id=123 \
    --rdzv_endpoint="node0:29500" \
    scripts/train.py \
    --config config/production.yaml

# Node 1 (worker)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=123 \
    --rdzv_endpoint="node0:29500" \
    scripts/train.py \
    --config config/production.yaml
```

### Checkpoint Management

```python
# In your training script, use GCS paths
checkpoint_path = "gs://YOUR_BUCKET_NAME/checkpoints/step_1000.pt"

# With tensorboard
tensorboard --logdir=gs://YOUR_BUCKET_NAME/logs
```

## Monitoring

### View GPU Utilization

```bash
# SSH into instance
nvidia-smi -l 1
```

### Cloud Logging

```bash
# View logs
gcloud logging read "resource.type=gce_instance" --limit=100

# Filter by training
gcloud logging read "resource.type=gce_instance AND logName:yadabo" --limit=100
```

## Cost Optimization

### Spot Instances (Preemptible)

```bash
# Create spot instance (80% cheaper)
gcloud compute instances create yadabo-spot \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --image-family=pytorch-latest \
    --image-project=deeplearning-platform-release \
    --provisioning-model=SPOT \
    --spot-instance-traits=preemptible
```

### Auto-Shutdown

```bash
# Create startup script to auto-shutdown after training
echo "#!/bin/bash
echo '0 6 * * * shutdown -h' | crontab -
" > startup.sh
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train.py --batch_size 16 --gradient_checkpointing

# Clear cache
nvidia-smi --gpu-reset
```

### Slow Training

```bash
# Check NCCL
python -c "import torch; torch.distributed.is_nccl_available()"

# Use mixed precision
python scripts/train.py --fp16
```

## Next Steps

1. Upload your dataset to Cloud Storage
2. Start with small model for testing
3. Scale up to larger models
4. Set up automated training with Cloud Functions
5. Enable model versioning with Vertex AI

## Resources

- [Google Cloud GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- [PyTorch on GCP](https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
