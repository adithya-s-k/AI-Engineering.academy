# Fine-tuning Llama 8-70B with Axolotl on Modal: Multi-GPU Training Made Simple

📄 **[View Complete Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/FinetuneLlamaAxolotlGPUModal.py)**

So you've trained nanoGPT from scratch and fine-tuned Gemma with Unsloth. Now let's go full beast mode - we're training Llama 8-70B across multiple GPUs. We're talking real production ML infrastructure here

And the crazy part? We're doing it all with Axolotl on Modal. No Kubernetes cluster to manage, no infrastructure nightmares. Just distributed training power.

## Why Axolotl?

I discovered Axolotl when I needed to fine-tune a 8-70B model and realized Unsloth wasn't built for multi-GPU setups. That's where Axolotl shines.

**What makes Axolotl special:**
- **Production-grade multi-GPU support** - Train across 2, 4, or even 8 GPUs without writing custom distributed code
- **YAML-based configs** - All your hyperparameters in one readable file. No more scattered parameters across Python code
- **Built-in DeepSpeed and FSDP** - The same tech Microsoft uses to train massive models, just works out of the box
- **Extensive model support** - Llama, Mistral, Qwen, you name it. Pre-configured recipes for all major models
- **Battle-tested** - Used by companies to train production models, not just a research toy

The thing is, when you're training a 8-70B model, you physically can't fit it on a single GPU. Even an A100-80GB can barely hold the model weights, let alone gradients and optimizer states. You NEED multi-GPU training.

Axolotl handles all the complexity: splitting the model across GPUs, synchronizing gradients, managing checkpoints. You just write a YAML file and hit run.

## What We're Building

This is a complete multi-GPU training pipeline with four independent stages:

1. **Preprocess datasets** - Tokenize and cache (on 1 GPU, because why waste money?)
2. **Multi-GPU training** - Fine-tune Llama 8-70B across 4 GPUs with LoRA
3. **Merge LoRA adapters** - Combine adapters with base model for deployment
4. **Inference** - Test your fine-tuned model

Here's the mental model:

```
┌──────────────────────┐
│  Preprocess Data     │  (1 GPU - $3.50/hr, run once)
│  Tokenize & Cache    │
└─────────┬────────────┘
          │
┌─────────▼────────────┐
│  Multi-GPU Training  │  (4× A100-80GB - $14/hr)
│  Llama 8-70B + LoRA    │  ← The beast mode part
└─────────┬────────────┘
          │
┌─────────▼────────────┐
│   Merge LoRA         │  (1 GPU - $3.50/hr, ~30 min)
│  Adapters → Model    │
└─────────┬────────────┘
          │
┌─────────▼────────────┐
│    Inference         │  (1 GPU - test your model)
└──────────────────────┘
```

Each stage is independent. Screw up training? Just re-run that step. Want to test different hyperparameters? Preprocessing is already cached.

## Getting Started

### Install Modal

You know the drill by now:

```bash
pip install modal
```

### Authenticate

```bash
modal setup
```

Or with API keys:

```bash
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### Set Up Your Secrets

For this, you'll need:
- **Hugging Face token** - Required for Llama models (they're gated)
- **Weights & Biases API key** - Optional but highly recommended for tracking training

**Create the Modal secret:**

```bash
modal secret create secrets-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx \
  WANDB_API_KEY=xxxxxxxxxxxxx
```

> **Note:** The script looks for a secret named `secrets-hf-wandb`. If you use a different name, update the code where it says `Secret.from_name("secrets-hf-wandb")`.

**Get your tokens:**
- HF token: [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
- W&B key: [wandb.ai/authorize](https://wandb.ai/authorize)

### Project Structure

This is even simpler than the previous tutorials:

```
ServerLessFinetuning/
├── FinetuneLlamaAxolotlGPUModal.py    # Everything lives here
└── .env                                # Optional: local secrets
```

One file. That's it. All your configuration, all your training stages, all your pipeline logic - in one clean Python file.

## Understanding Multi-GPU Training

Before we dive into code, let's understand why we even need multiple GPUs.

### The Memory Problem

Here's the brutal math:

| Model Size | Parameters | FP16 Weights | Training Memory | Fits on Single GPU? |
|------------|-----------|--------------|-----------------|-------------------|
| Llama 3-8B | 8 billion | ~16GB | ~40GB | ✓ (A100-80GB) |
| Llama 3-8-70B | 70 billion | ~140GB | ~280GB | ✗ (Impossible!) |
| Llama 3-405B | 405 billion | ~810GB | ~1.6TB | ✗ (Very impossible!) |

Training memory = model weights + gradients + optimizer states + activations. Roughly 2-3x the model size.

A single A100-80GB has... 80GB. You literally cannot fit a 8-70B model for training, even with quantization.

### The Multi-GPU Solution

There are several strategies to distribute a model across GPUs:

1. **Data Parallelism (DP):** Copy the full model to each GPU, split the batch across them. Great for small models, useless for 8-70B.

2. **Tensor Parallelism (TP):** Split individual layers across GPUs. Each GPU has part of each attention head, part of each MLP. Complex but efficient.

3. **Pipeline Parallelism (PP):** Different GPUs process different layers. GPU 0 has layers 1-20, GPU 1 has 21-40, etc. Simple but can have bubble time.

4. **FSDP (Fully Sharded Data Parallel):** The modern approach. Shard everything - model parameters, gradients, optimizer states. Each GPU only keeps what it needs, fetches the rest when needed.

**The beauty of Axolotl?** You don't have to think about this. It uses HuggingFace Accelerate under the hood, which figures out the best strategy automatically. You just say "use 4 GPUs" and it handles the rest.

## Configuration Overview

Let's start with the basics:

```python
from modal import App, Image as ModalImage, Volume, Secret

# Create the Modal app
app = App("Finetuned_Llama_70b_Axolotl_MultiGPU")

# Create persistent storage
exp_volume = Volume.from_name("Finetuned_Llama_70b_Axolotl", create_if_missing=True)

# Mount the volume at /data in all containers
VOLUME_CONFIG = {
    "/data": exp_volume,
}

# Load secrets
huggingface_secret = Secret.from_name("secrets-hf-wandb")
```

**What's happening:**
- **Volume:** All our data lives here - preprocessed datasets, checkpoints, final models. Persists across runs.
- **Secrets:** Injected as environment variables in our containers. Clean and secure.

### Configuration Constants

```python
# Time constants
HOURS = 60 * 60  # Makes timeouts readable

# GPU Configuration
GPU_TYPE = "a100-80gb"  # Can be: a100-80gb, a100-40gb, l40s, etc.

# Training Configuration
WANDB_PROJECT_DEFAULT = "Llama-70b-MultiGPU-finetune"
```

**GPU type considerations:**

For **8B models:**
- L40S works great (~$1/hr)
- A100-40GB for comfort ($2.50/hr)

For **8-70B models:**
- A100-80GB is required ($3.50/hr per GPU)
- You'll need 4-8 of them ($14-28/hr total)

For **405B models:**
- 8× A100-80GB minimum ($28/hr)
- Or use H100s, B100s

## Building the Axolotl Image

This is where things get interesting. We need a container with CUDA, PyTorch, Axolotl, DeepSpeed, Flash Attention... the works.

### CUDA Base Configuration

```python
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"
```

**Why "devel"?**

Flash Attention (which makes training way faster) needs to compile CUDA code during installation. The `runtime` image doesn't include the CUDA compiler (`nvcc`), so you'll get cryptic errors.

The `devel` image includes the full CUDA toolkit. It's bigger, but it Just Works™.

### Complete Image Definition

```python
AXOLOTL_IMAGE = (
    # Start with NVIDIA's official CUDA image
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")

    # Install system dependencies
    .apt_install("git", "build-essential")

    # Install PyTorch first
    .uv_pip_install([
        "torch",
        "torchvision",
        "torchaudio",
    ])

    # Install base dependencies for Axolotl
    .run_commands(
        "uv pip install --no-deps -U packaging setuptools wheel ninja --system"
    )

    # Install Axolotl with DeepSpeed support
    .run_commands("uv pip install --no-build-isolation axolotl[deepspeed] --system")

    # Install Flash Attention (this takes a while to compile)
    .run_commands(
        "UV_NO_BUILD_ISOLATION=1 uv pip install flash-attn --no-build-isolation --system"
    )

    # Set environment variables
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Fast model downloads
        "HF_HOME": "/data/.cache",          # Cache in volume
    })
)
```

**Key points:**

1. **Installation order matters:** Base deps → Axolotl → Flash Attention. Do it wrong and you'll have dependency hell.

2. **`--no-build-isolation`:** Required for Flash Attention. Don't ask me why, it's just how flash-attn works.

3. **`HF_HUB_ENABLE_HF_TRANSFER`:** Enables parallel downloads from HuggingFace. Can be 5-10x faster for large models.

4. **Cache in volume:** All HuggingFace downloads go to `/data/.cache`, which persists. Download once, use forever.

> **⏰ Build time warning:** The first build takes 20-30 minutes because Flash Attention compiles from source. It's compiling optimized CUDA kernels for your GPU architecture. Go grab lunch. But here's the magic - Modal caches this image. Every subsequent run? Instant startup.

### Alternative: Official Axolotl Image

Axolotl provides an official Docker image, but I don't use it:

```python
# This works, but I don't recommend it:
# AXOLOTL_IMAGE = ModalImage.from_registry(
#     "axolotlai/axolotl-cloud:main-latest", add_python="3.12"
# ).env({
#     "JUPYTER_ENABLE_LAB": "no",
#     "JUPYTER_TOKEN": "",
#     "HF_HOME": "/data/.cache",
# })
```

**Why custom image?**
- Official image auto-starts JupyterLab, which we don't need on Modal
- Custom image is lighter and more predictable
- Full control over versions (important for reproducibility)

## Training Configuration with YAML

Here's where Axolotl really shines. All your configuration goes in a YAML file. No scattered parameters, no magic constants buried in code. Everything in one place.

### Complete Training Config

```python
TRAIN_CONFIG_YAML = f"""
base_model: NousResearch/Meta-Llama-3-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

# Optional: Automatically upload to HuggingFace Hub
# hub_model_id: username/custom_model_name

load_in_8bit: true
load_in_4bit: false

chat_template: llama3
datasets:
  - path: fozziethebeat/alpaca_messages_2k_test
    type: chat_template

dataset_prepared_path: /data/prepared_datasets/alpaca_2k
val_set_size: 0.05
output_dir: /data/outputs/lora-out

sequence_len: 4096
sample_packing: false

# LoRA Configuration
adapter: lora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

# Weights & Biases
wandb_project: {WANDB_PROJECT_DEFAULT}
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

# Training Hyperparameters
gradient_accumulation_steps: 4
micro_batch_size: 8
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
tf32: false

# Memory Optimizations
gradient_checkpointing: true
resume_from_checkpoint:
logging_steps: 1
flash_attention: true

warmup_ratio: 0.1
evals_per_epoch: 4
saves_per_epoch: 4
weight_decay: 0.0

special_tokens:
   pad_token: <|end_of_text|>
"""
```

Let me break down the important parts:

### Model Settings

```yaml
base_model: NousResearch/Meta-Llama-3-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
```

This is configured for Llama 3-8B (for testing). When you're ready for the real deal:

```yaml
base_model: meta-llama/Meta-Llama-3-8-70B-Instruct
```

**For other models:**
```yaml
# Mistral 7B
base_model: mistralai/Mistral-7B-Instruct-v0.2
model_type: MistralForCausalLM

# Qwen 72B
base_model: Qwen/Qwen2.5-72B-Instruct
model_type: Qwen2ForCausalLM
```

### Quantization Settings

```yaml
load_in_8bit: true   # Reduces memory by ~50%
load_in_4bit: false  # Reduces memory by ~75%
```

**My recommendations:**
- **8B models:** `load_in_8bit: false` (use full precision, you have the memory)
- **8-70B models:** `load_in_8bit: true` (essential to fit on 4 GPUs)
- **405B models:** `load_in_4bit: true` (required, even with 8 GPUs)

The quality hit from 8-bit is minimal. The quality hit from 4-bit is noticeable but acceptable.

### Dataset Configuration

```yaml
datasets:
  - path: fozziethebeat/alpaca_messages_2k_test
    type: chat_template

dataset_prepared_path: /data/prepared_datasets/alpaca_2k
val_set_size: 0.05  # 5% for validation
```

**Multiple datasets:**
```yaml
datasets:
  - path: dataset1/name
    type: chat_template
  - path: dataset2/name
    type: alpaca
    split: train
```

Axolotl supports many formats: `chat_template`, `alpaca`, `sharegpt`, `completion`, etc. Check the [Axolotl docs](https://docs.axolotl.ai/) for the full list.

### LoRA Parameters

```yaml
adapter: lora
lora_r: 32           # Rank (higher = more capacity, slower training)
lora_alpha: 16       # Scaling factor (usually r/2 or r)
lora_dropout: 0.05   # Regularization
lora_target_linear: true  # Apply to all linear layers
```

**LoRA rank guidelines:**

For **8B models:**
```yaml
lora_r: 32
lora_alpha: 64
```

For **8-70B models (high quality):**
```yaml
lora_r: 64
lora_alpha: 128
```

For **faster training:**
```yaml
lora_r: 16
lora_alpha: 32
```

Higher rank = more capacity to learn, but slower training and larger adapters.

### Training Hyperparameters

```yaml
micro_batch_size: 8              # Batch size per GPU
gradient_accumulation_steps: 4   # Steps before updating weights
num_epochs: 4
learning_rate: 0.0002
optimizer: adamw_bnb_8bit        # 8-bit Adam (saves memory!)
lr_scheduler: cosine
```

**Effective batch size calculation:**
```
Effective Batch = micro_batch_size × gradient_accumulation_steps × num_gpus
                = 8 × 4 × 4
                = 128
```

With 4 GPUs, each does batch size 8, accumulates over 4 steps, so effectively you're training with batch size 128.

### Memory Optimizations

```yaml
gradient_checkpointing: true  # Trade compute for memory (essential!)
flash_attention: true         # Faster, more memory-efficient attention
bf16: auto                    # Use bfloat16 if GPU supports it
```

**Gradient checkpointing** is critical for large models. It recomputes activations during backward pass instead of storing them. Uses ~40% less memory at the cost of ~20% slower training. Totally worth it.

**Flash Attention** is a must-have. It's an optimized attention implementation that's both faster AND uses less memory. Win-win.

## Helper Function: Write Config to Volume

Before we get to the training stages, here's a helper function that all stages use:

```python
def write_config_to_volume(
    train_config_yaml: str,
    config_path: str = "/data/config.yml",
    update_paths: bool = True,
) -> dict:
    """Write YAML configuration to volume with optional path updates."""
    import os
    import yaml

    # Parse YAML string into dict
    config_dict = yaml.safe_load(train_config_yaml)

    # Update paths to use volume instead of local dirs
    if update_paths and "output_dir" in config_dict:
        config_dict["output_dir"] = config_dict["output_dir"].replace(
            "./outputs", "/data/outputs"
        )

    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Write to volume
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Commit so it persists
    exp_volume.commit()

    return config_dict
```

**What it does:**
1. Converts YAML string to dict (for inspection)
2. Updates paths to use `/data` volume (so outputs persist)
3. Writes config to volume
4. Commits volume (critical!)

This keeps our config in one place and ensures all stages use the same configuration.

## Stage 1: Dataset Preprocessing

Alright, let's get to the actual pipeline. First stage: preprocessing.

```python
# GPU Configuration for preprocessing (single GPU is fine)
PREPROCESS_NUM_GPUS = 1
PREPROCESS_GPU_CONFIG = f"{GPU_TYPE}:{PREPROCESS_NUM_GPUS}"

@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
    gpu=PREPROCESS_GPU_CONFIG,  # Just 1 GPU
)
def process_datasets(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
):
    """Preprocess and tokenize dataset before training using Axolotl."""
    import os
    import subprocess

    # Set HF token
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Write config to volume
    config_dict = write_config_to_volume(train_config_yaml, config_path, True)
    exp_volume.commit()

    print("Starting dataset preprocessing...")

    try:
        # Run Axolotl preprocessing
        subprocess.run(["axolotl", "preprocess", config_path], check=True)
        print("✓ Preprocessing completed")

        # Commit preprocessed data
        exp_volume.commit()

        return {
            "status": "completed",
            "config_path": config_path,
            "preprocessed_data_path": config_dict.get("dataset_prepared_path"),
            "output_dir": config_dict.get("output_dir"),
        }
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Preprocessing failed: {e}")
```

**What happens during preprocessing:**

1. **Download dataset** from HuggingFace (or load from cache)
2. **Apply chat template** to format conversations correctly
3. **Tokenize everything** using the model's tokenizer
4. **Save to disk** at `dataset_prepared_path`
5. **Split train/val** based on `val_set_size`

**Why preprocess separately?**

You might think "why not just preprocess during training?" Here's why this is better:

1. **Cost savings:** Preprocessing doesn't need 4 GPUs. Why pay $14/hr when you can pay $3.50/hr?

2. **Reusability:** Preprocess once, train multiple times with different hyperparameters. Huge time saver when experimenting.

3. **Debugging:** If preprocessing fails, you know immediately. Not after 3 hours of training setup.

4. **Visibility:** You can inspect the preprocessed data to make sure it looks right.

**Run it:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::process_datasets
```

First run downloads the dataset and tokenizes everything. Takes 10-30 minutes depending on dataset size. Subsequent runs? Instant, because it's cached.

## Stage 2: Multi-GPU Training

Here's where the magic happens. We're training Llama across multiple GPUs with Accelerate.

### GPU Configuration

```python
# GPU Configuration for training (2-8 GPUs)
TRAIN_NUM_GPUS = 4  # Can be adjusted from 2 to 8
TRAIN_GPU_CONFIG = f"{GPU_TYPE}:{TRAIN_NUM_GPUS}"
```

**Scaling guidelines:**

| Model | Min GPUs | Recommended | GPU Type | Cost/hr |
|-------|----------|-------------|----------|---------|
| Llama 3-8B | 1 | 1 | A100-40GB | $2.50 |
| Llama 3-13B | 1 | 2 | A100-40GB | $5.00 |
| Llama 3-8-70B | 2 | 4 | A100-80GB | $14.00 |
| Llama 3-405B | 8 | 8 | A100-80GB | $28.00 |

### Training Function

```python
@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
    gpu=TRAIN_GPU_CONFIG,  # e.g., "a100-80gb:4"
)
def train_model(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
):
    """
    Train or fine-tune a model using Axolotl with multi-GPU support.
    Uses accelerate for multi-GPU training.
    """
    import os
    import subprocess

    # Set up environment
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_DEFAULT

    # Write config to volume
    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    exp_volume.commit()

    # Build accelerate command for multi-GPU training
    print(f"Starting training with {TRAIN_NUM_GPUS} GPUs...")

    cmd = [
        "accelerate",
        "launch",
        "--multi_gpu",                        # Enable multi-GPU mode
        "--num_processes", str(TRAIN_NUM_GPUS),  # Number of GPUs
        "--num_machines", "1",                # Single machine (Modal handles this)
        "--mixed_precision", "bf16",          # Use bfloat16 for speed
        "--dynamo_backend", "no",             # Disable torch.compile (stability)
        "-m", "axolotl.cli.train",            # Run Axolotl training
        config_path,                          # Path to our YAML config
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✓ Training completed")

        # Commit trained model to volume
        exp_volume.commit()

        return {
            "status": "completed",
            "config_path": config_path,
            "output_dir": config_dict.get("output_dir"),
            "base_model": config_dict.get("base_model"),
            "num_gpus": TRAIN_NUM_GPUS,
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Training failed: {e}")
```

### Understanding the Accelerate Command

Let me break down what this command does:

```bash
accelerate launch \
  --multi_gpu \                     # Enable multi-GPU distributed training
  --num_processes 4 \               # Use 4 GPUs (one process per GPU)
  --num_machines 1 \                # Single machine (Modal provides this)
  --mixed_precision bf16 \          # Use bfloat16 for memory and speed
  --dynamo_backend no \             # Disable torch.compile (causes issues)
  -m axolotl.cli.train \            # Run Axolotl's training module
  /data/config.yml                  # Our YAML configuration
```

**What Accelerate does for you:**

1. **Spawns one process per GPU** - Each GPU gets its own Python process
2. **Initializes distributed backend** - Sets up NCCL for GPU communication
3. **Shards the model** - Splits model parameters across GPUs based on available memory
4. **Synchronizes gradients** - All-reduce operation after backward pass
5. **Manages checkpointing** - Handles saving/loading distributed checkpoints

You write normal PyTorch code (which Axolotl already did), Accelerate makes it distributed. It's magical.

### Run Training

**Basic run (test with 8B model first):**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**For actual 8-70B training, edit the YAML:**

```python
TRAIN_CONFIG_YAML = f"""
base_model: meta-llama/Meta-Llama-3-8-70B-Instruct
load_in_8bit: true
lora_r: 64
lora_alpha: 128
micro_batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
# ... rest of config
"""
```

Then run:

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**Monitor your training:**

1. **Modal Dashboard:** Click the URL in the terminal for real-time logs and GPU utilization
2. **Weights & Biases:** Go to `wandb.ai/<username>/Llama-70b-MultiGPU-finetune` for beautiful charts
3. **Check GPU usage:** All 4 GPUs should be near 100% utilization

> **💰 Cost Alert:** 4× A100-80GB costs ~$14/hour. A 3-hour training run = $42. A 10-hour run = $140. This is why we preprocess separately and test with small models first!

**Training checkpoints:**

Axolotl automatically saves checkpoints to `/data/outputs/lora-out/checkpoint-{step}`. If training crashes, resume with:

```yaml
resume_from_checkpoint: /data/outputs/lora-out/checkpoint-1000
```

## Stage 3: Merge LoRA Adapters

After training, you have LoRA adapters (~100MB) that work with the base model. For deployment, it's easier to merge them into a single model.

```python
# GPU Configuration for merging (single GPU is fine)
MERGE_NUM_GPUS = 1
MERGE_GPU_CONFIG = f"{GPU_TYPE}:{MERGE_NUM_GPUS}"

@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=4 * HOURS,
    gpu=MERGE_GPU_CONFIG,
)
def merge_lora(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
    lora_model_dir: str = None,
):
    """Merge trained LoRA adapters into the base model."""
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Write config
    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    exp_volume.commit()

    # Build merge command
    print("Starting LoRA merge...")
    cmd = ["axolotl", "merge-lora", config_path]

    if lora_model_dir:
        cmd.extend(["--lora-model-dir", lora_model_dir])

    try:
        subprocess.run(cmd, check=True)
        print("✓ LoRA merge completed")

        # Commit merged model
        exp_volume.commit()

        return {
            "status": "completed",
            "config_path": config_path,
            "output_dir": config_dict.get("output_dir"),
            "lora_model_dir": lora_model_dir or config_dict.get("lora_model_dir"),
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LoRA merge failed: {e}")
```

**What merging does:**

1. Loads base model weights
2. Loads LoRA adapter weights
3. Applies the LoRA transformations to base model
4. Saves the combined model

**Why only 1 GPU?**

Merging is sequential - you're just doing matrix operations to combine weights. Doesn't benefit from multiple GPUs. Save the money.

**Run it:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::merge_lora
```

**Specify custom LoRA directory:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::merge_lora \
  --lora-model-dir="/data/outputs/lora-out"
```

Merging takes 15-45 minutes depending on model size. For 8-70B, expect ~30 minutes.

## Stage 4: Inference

Let's test your fine-tuned model!

```python
# GPU Configuration for inference (single GPU)
INFERENCE_NUM_GPUS = 1
INFERENCE_GPU_CONFIG = f"{GPU_TYPE}:{INFERENCE_NUM_GPUS}"

@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
    gpu=INFERENCE_GPU_CONFIG,
)
def run_inference(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
    prompt: str = "Hello, how are you?",
    lora_model_dir: str = None,
    base_model: str = None,
):
    """Run inference using the trained model."""
    import os
    import subprocess
    import tempfile

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Write config
    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    print("Starting inference...")
    print(f"Prompt: {prompt}")
    print("-" * 80)

    # Build inference command
    cmd = ["axolotl", "inference", config_path]

    if lora_model_dir:
        cmd.extend(["--lora-model-dir", lora_model_dir])
    if base_model:
        cmd.extend(["--base-model", base_model])

    # Write prompt to temp file and pipe it
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(prompt)
            prompt_file = f.name

        # Run inference with prompt from file
        with open(prompt_file, "r") as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
                check=True,
            )

        print("✓ Inference completed")
        print("\n" + "=" * 80)
        print("MODEL OUTPUT:")
        print("=" * 80)
        print(result.stdout)
        print("=" * 80)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        return {
            "status": "completed",
            "prompt": prompt,
            "output": result.stdout,
            "model": base_model or config_dict.get("base_model"),
        }

    except subprocess.CalledProcessError as e:
        print(f"Error output: {e.stderr}")
        print(f"Command output: {e.stdout}")
        raise RuntimeError(f"Inference failed: {e}")
    finally:
        # Clean up temp file
        import os as os_module
        if "prompt_file" in locals():
            os_module.unlink(prompt_file)
```

**Run inference:**

**With LoRA adapters (before merging):**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::run_inference \
  --prompt="Explain quantum computing in simple terms." \
  --lora-model-dir="/data/outputs/lora-out"
```

**With merged model:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::run_inference \
  --prompt="Write a poem about machine learning." \
  --base-model="/data/outputs/lora-out-merged"
```

**Test multiple prompts:**

Create a Python script to batch test:

```python
import modal

app = modal.App.lookup("Finetuned_Llama_70b_Axolotl_MultiGPU")
run_inference = modal.Function.lookup(app.name, "run_inference")

prompts = [
    "Explain gradient descent.",
    "Write Python code to implement binary search.",
    "What are the benefits of transformer architecture?",
]

for prompt in prompts:
    result = run_inference.remote(prompt=prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {result['output']}\n")
    print("-" * 80)
```

## Complete Workflow Example

Let me walk you through how I actually use this for a real project.

### 1. Customize Configuration

First, I edit the YAML config in the script:

```python
TRAIN_CONFIG_YAML = f"""
base_model: meta-llama/Meta-Llama-3-8-70B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: true

chat_template: llama3
datasets:
  - path: my-username/my-custom-dataset
    type: chat_template

dataset_prepared_path: /data/prepared_datasets/my_data
output_dir: /data/outputs/llama70b-custom

sequence_len: 8192  # Longer context
lora_r: 64
lora_alpha: 128

gradient_accumulation_steps: 2
micro_batch_size: 4
num_epochs: 3
learning_rate: 0.0001

wandb_project: {WANDB_PROJECT_DEFAULT}

# Everything else stays the same
gradient_checkpointing: true
flash_attention: true
bf16: auto
"""
```

### 2. Preprocess Dataset

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::process_datasets
```

**Expected time:** 30 min - 2 hours (depends on dataset size)
**Cost:** ~$2-5 (1 GPU for preprocessing)

Grab a coffee while this runs.

### 3. Test Training (Small Sanity Check)

Before burning $100 on a full training run, I test with 1 epoch:

```python
# Temporarily edit YAML:
num_epochs: 1
```

Then:

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**Expected time:** 30-60 minutes
**Cost:** ~$7-14

This catches configuration errors, memory issues, etc. If this works, the full run will work.

### 4. Full Training

Restore the full config and run:

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**Expected time:** 3-10 hours (depends on dataset size)
**Cost:** $40-150

Now you wait. Monitor W&B to make sure loss is going down. Check Modal dashboard to verify all GPUs are utilized.

Go touch grass. This is running on Modal's infrastructure, not your machine.

### 5. Merge LoRA

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::merge_lora
```

**Expected time:** 30-45 minutes
**Cost:** ~$2-3

### 6. Test Inference

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::run_inference \
  --prompt="Test my fine-tuned 8-70B model with this prompt."
```

**Expected time:** 1-2 minutes
**Cost:** ~$0.10

### 7. Push to Hub (Optional)

Want to share your model? Add this to the YAML config:

```yaml
hub_model_id: your-username/llama-70b-custom-finetuned
```

Axolotl automatically pushes to HuggingFace Hub during training.

**Total cost for full pipeline:** $50-200 depending on dataset size and number of epochs.

**Total time:** 1 day (mostly waiting for training)

Compare this to managing your own 4× A100-80GB cluster... yeah, Modal wins.

## Advanced Tips and Tricks

### Multi-Dataset Training

Train on multiple datasets simultaneously:

```yaml
datasets:
  - path: dataset1/name
    type: chat_template
  - path: dataset2/name
    type: alpaca
  - path: dataset3/name
    type: sharegpt
```

Axolotl combines them automatically.

### Checkpoint Management

```yaml
saves_per_epoch: 4              # Save 4 times per epoch
save_total_limit: 10            # Keep only 10 most recent checkpoints
resume_from_checkpoint: /data/outputs/lora-out/checkpoint-1000
```

**Pro tip:** If training crashes or you want to tweak learning rate halfway through, just resume from a checkpoint.

### Custom Evaluation

```yaml
val_set_size: 0.1           # 10% for validation
evals_per_epoch: 10         # Evaluate 10 times per epoch
```

Watch validation loss in W&B to catch overfitting.

### Optimizer Tuning

```yaml
optimizer: adamw_bnb_8bit      # Options: adamw_torch, adamw_bnb_8bit, adafactor
lr_scheduler: cosine           # Options: linear, cosine, constant
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
```

I usually stick with `adamw_bnb_8bit` (saves memory) and `cosine` scheduler (smooth learning rate decay).

## Hyperparameter Tuning Guide

### For Llama 3-8B (Single GPU)

```yaml
micro_batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 0.0003
lora_r: 32
lora_alpha: 64
num_epochs: 3
load_in_8bit: false  # Can use full precision
```

**GPU:** 1× A100-40GB
**Cost:** ~$2.50/hr
**Training time:** 2-4 hours

### For Llama 3-8-70B (Multi-GPU)

```yaml
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0001
lora_r: 64
lora_alpha: 128
num_epochs: 2-3
load_in_8bit: true  # Essential
```

**GPU:** 4× A100-80GB
**Cost:** ~$14/hr
**Training time:** 4-10 hours

### For Maximum Speed

```yaml
flash_attention: true
bf16: auto
gradient_checkpointing: true
sample_packing: true              # Pack multiple samples per sequence
pad_to_sequence_len: true
optimizer: adamw_bnb_8bit
```

### For Maximum Quality

```yaml
learning_rate: 0.00005            # Lower LR = more stable
num_epochs: 5                     # More epochs
warmup_ratio: 0.2                 # Longer warmup
lora_r: 128                       # Higher capacity
lora_alpha: 256
micro_batch_size: 2               # Smaller batches if you have memory
```

Quality vs. Speed is always a tradeoff. Experiment!

## Common Issues and Solutions

### "Out of Memory" During Training

**Error:** `CUDA out of memory`

**Solutions (in order of preference):**

1. **Reduce batch size:**
   ```yaml
   micro_batch_size: 2
   gradient_accumulation_steps: 8  # Keep effective batch size the same
   ```

2. **Enable gradient checkpointing** (if not already):
   ```yaml
   gradient_checkpointing: true
   ```

3. **Use quantization:**
   ```yaml
   load_in_8bit: true
   ```

4. **Add more GPUs:**
   ```python
   TRAIN_NUM_GPUS = 8
   ```

5. **Reduce sequence length:**
   ```yaml
   sequence_len: 2048
   ```

### Training Loss Not Decreasing

**Symptoms:** Loss stays flat or increases

**Solutions:**

1. **Check your learning rate** - might be too low:
   ```yaml
   learning_rate: 0.0003
   ```

2. **Verify data quality:**
   - Load a few samples from preprocessed data
   - Make sure they look correct

3. **Increase LoRA rank:**
   ```yaml
   lora_r: 64
   lora_alpha: 128
   ```

4. **Train longer:**
   ```yaml
   num_epochs: 5
   ```

5. **Check for data leakage** - validation set might be in training set

### Preprocessing Hangs or Fails

**Solution:**

Test dataset access locally first:

```python
from datasets import load_dataset
dataset = load_dataset("your/dataset", split="train")
print(f"Loaded {len(dataset)} samples")
print(dataset[0])
```

If that works, ensure your HF token is in Modal secrets.

### Multi-GPU Not Working

**Symptoms:** Training only uses 1 GPU

**Debug steps:**

1. **Verify GPU count:**
   ```python
   TRAIN_NUM_GPUS = 4  # Check this!
   ```

2. **Check GPU utilization in Modal dashboard** - should see all 4 GPUs active

3. **Add debug prints:**
   ```python
   import torch
   print(f"GPUs available: {torch.cuda.device_count()}")
   ```

### "Secret not found"

**Error:** `Modal Secret "secrets-hf-wandb" not found`

**Solution:**

```bash
modal secret create secrets-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxx \
  WANDB_API_KEY=xxx
```

Or update the script to use your secret name:

```python
huggingface_secret = Secret.from_name("my-secret-name")
```

## Cost Optimization Strategies

### 1. Always Preprocess Separately

**Bad (expensive):**
```bash
# This runs preprocessing on 4 GPUs!
modal run script.py::train_model
```

**Good (cheap):**
```bash
# Preprocess on 1 GPU
modal run script.py::process_datasets  # $3.50/hr

# Train on 4 GPUs
modal run script.py::train_model      # $14/hr
```

**Savings:** ~$10/hr during preprocessing (which can take 1-2 hours)

### 2. Test with Smaller Models First

```python
# Test with 8B first
base_model: NousResearch/Meta-Llama-3-8B-Instruct
TRAIN_NUM_GPUS = 1

# Then scale to 8-70B
base_model: meta-llama/Meta-Llama-3-8-70B-Instruct
TRAIN_NUM_GPUS = 4
```

Catch bugs on the cheap model, then run the expensive one.

### 3. Use Smaller GPUs for Testing

```python
# Testing phase
GPU_TYPE = "l40s"  # ~$1/hr

# Production phase
GPU_TYPE = "a100-80gb"  # ~$3.50/hr
```

### 4. Limit Test Datasets

For testing, use a small subset:

```yaml
datasets:
  - path: dataset/name
    type: chat_template
    num_samples: 100  # Just 100 samples for testing
```

Or edit the dataset on HuggingFace to create a "mini" version.

### 5. Smart Checkpointing

```yaml
saves_per_epoch: 2           # Don't checkpoint too often
save_total_limit: 5          # Delete old checkpoints
```

Volume storage is free up to 50GB, but good practice for huge models.

### 6. Resume from Checkpoint

If training fails or you want to try different hyperparameters:

```yaml
resume_from_checkpoint: /data/outputs/lora-out/checkpoint-1000
```

**Don't start from scratch!** You've already paid for those first 1000 steps.

## Monitoring and Debugging

### Real-time Logs

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
# Click the URL in output to open Modal dashboard
```

**Dashboard shows:**
- Real-time logs (stdout/stderr)
- GPU utilization per GPU (should be ~95-100%)
- Memory usage per GPU
- Cost accumulation ($$$ ticking up)

### Weights & Biases

Go to `wandb.ai/<username>/Llama-70b-MultiGPU-finetune`

**Charts to watch:**
- **Training loss** - should decrease smoothly
- **Validation loss** - should decrease, but slower than training
- **Learning rate** - should follow the schedule (cosine decay)
- **GPU utilization** - should be high

**If training loss decreases but validation loss increases:** You're overfitting. Reduce epochs or add regularization.

### Check Preprocessed Data

```bash
modal volume ls Finetuned_Llama_70b_Axolotl /data/prepared_datasets
```

Lists preprocessed files. You can download and inspect:

```bash
modal volume get Finetuned_Llama_70b_Axolotl \
  /data/prepared_datasets/alpaca_2k \
  ./local_data
```

### Download Checkpoints

```bash
# List checkpoints
modal volume ls Finetuned_Llama_70b_Axolotl /data/outputs/lora-out

# Download specific checkpoint
modal volume get Finetuned_Llama_70b_Axolotl \
  /data/outputs/lora-out/checkpoint-1000 \
  ./local_checkpoint
```

Useful for local testing or pushing to HuggingFace manually.

## Scaling to 8 GPUs (For the Brave)

For massive models like Llama 3-405B or Mixtral 8×22B, you need 8 GPUs.

### Update Configuration

```python
TRAIN_NUM_GPUS = 8
GPU_TYPE = "a100-80gb"
```

### Update YAML for Memory

```yaml
micro_batch_size: 2
gradient_accumulation_steps: 2
load_in_8bit: true  # Or even 4bit

# Effective batch: 2 × 2 × 8 = 32
```

### Enable DeepSpeed ZeRO Stage 3 (Optional)

For maximum memory efficiency, create `deepspeed_zero3.json`:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  },
  "fp16": {"enabled": true},
  "train_micro_batch_size_per_gpu": 2
}
```

Add to YAML:

```yaml
deepspeed: /path/to/deepspeed_zero3.json
```

DeepSpeed ZeRO Stage 3 shards everything - parameters, gradients, optimizer states - across GPUs. Even offloads to CPU when needed. Insanely memory efficient but adds communication overhead.

**Cost:** 8× A100-80GB = ~$28/hour

A 10-hour training run = $280. Make sure you really need 8 GPUs!

## What's Next?

You've built a production-grade multi-GPU training pipeline. Here's what you can do next:

1. **Custom datasets:** Format your own data for Axolotl (see [Axolotl dataset docs](https://docs.axolotl.ai/))

2. **Advanced LoRA variants:** Try QLoRA (4-bit quantized LoRA) or DoRA (decomposed LoRA)

3. **Full fine-tuning:** Remove `adapter: lora` and train all parameters (requires way more memory)

4. **Evaluation:** Add custom evaluation metrics beyond just loss

5. **Deployment:** Serve your model with vLLM (see the Gemma tutorial for details)

6. **Multi-node training:** Scale beyond 8 GPUs using Modal's multi-node support (advanced!)

## Resources

- **[Axolotl Documentation](https://docs.axolotl.ai/)** - Official docs with dataset formats and advanced configs
- **[Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)** - Source code and issue tracker
- **[Axolotl Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples)** - Pre-built configs for tons of models
- **[Modal Documentation](https://modal.com/docs)** - Everything about Modal
- **[HuggingFace Accelerate](https://huggingface.co/docs/accelerate)** - Deep dive into distributed training
- **[DeepSpeed](https://www.deepspeed.ai/)** - For ZeRO optimization details

## Wrapping Up

You just built what most companies call "ML infrastructure":
- Multi-GPU distributed training
- Automatic checkpointing and resumption
- Experiment tracking with W&B
- Model versioning on HuggingFace
- Cost-optimized preprocessing pipeline

All in one Python file, running on Modal. No Kubernetes, no Docker nightmares, no spending weeks setting up infrastructure.

The Unsloth tutorial showed you highly optimized single-GPU training. This tutorial showed you how to scale beyond that - training models that literally don't fit on a single GPU.

This is how real ML teams train models nowadays. Not on local machines, not on hand-managed clusters. On serverless infrastructure like Modal, where you pay by the second and scale infinitely.

The YAML configuration approach is especially powerful. Need to try different learning rates? Just edit one line and re-run. Want to train on a different dataset? Change one parameter. Everything is reproducible, everything is versioned, everything is clean.

Got questions? Hit me up on Twitter [@adithya_s_k](https://x.com/adithya_s_k)!

Now go train that 8-70B model and show the world what you built. 🚀
