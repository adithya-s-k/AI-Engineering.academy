# Multi-GPU Fine-tuning with Axolotl on Modal: Training Llama 70B at Scale

This advanced tutorial covers distributed multi-GPU training using Axolotl on Modal. We'll fine-tune Llama models (8B to 70B) across 2-8 GPUs using DeepSpeed and FSDP for maximum efficiency.

## Why Axolotl?

**Axolotl** is a production-grade fine-tuning framework with:
- **Multi-GPU support** via DeepSpeed, FSDP, and Accelerate
- **YAML-based configuration** for reproducibility
- **Pre-built recipes** for popular models
- **Extensive customization** options

**Modal + Axolotl** enables training massive models without infrastructure headaches.

## What You'll Learn

- Setting up multi-GPU training environments
- YAML configuration for Axolotl
- Data preprocessing pipelines
- Distributed training with Accelerate
- LoRA merging and inference
- Cost-effective scaling strategies

---

## Prerequisites

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

Or with API keys:

```bash
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### 3. Set Up Secrets

You'll need:
- **Hugging Face token** (for gated models like Llama)
- **Weights & Biases API key** (optional, recommended)

#### Create Modal Secret

```bash
modal secret create adithya-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx \
  WANDB_API_KEY=xxxxxxxxxxxxx
```

> **Note:** The script references `Secret.from_name("adithya-hf-wandb")`. Either create this secret or modify the code to use your secret name.

---

## Project Structure

```
ServerLessFinetuning/
├── FinetuneLlamaAxolotlGPUModal.py    # Complete multi-GPU training pipeline
└── .env                                # Optional: local secrets
```

No additional files needed—everything is in one Python script!

---

## Understanding Multi-GPU Training

### Why Multiple GPUs?

| Model Size | Min VRAM | Single GPU | Multi-GPU Solution |
|------------|----------|------------|-------------------|
| Llama 3-8B | ~16GB | L40S, A100-40GB | Not needed |
| Llama 3-70B | ~140GB | Impossible | 4× A100-80GB |
| Llama 3-405B | ~810GB | Impossible | 8× A100-80GB |

**Multi-GPU strategies:**
1. **Data Parallelism:** Split batches across GPUs
2. **Tensor Parallelism:** Split model layers across GPUs
3. **Pipeline Parallelism:** Different GPUs process different layers
4. **FSDP (Fully Sharded Data Parallel):** Shard model parameters and optimizer states

Axolotl + Accelerate handle this automatically!

---

## Configuration Overview

### App, Volume, and Secrets

```python
from modal import App, Image as ModalImage, Volume, Secret

app = App("Finetuned_Llama_70b_Axolotl_MultiGPU")

# Persistent storage
exp_volume = Volume.from_name("Finetuned_Llama_70b_Axolotl", create_if_missing=True)
VOLUME_CONFIG = {
    "/data": exp_volume,
}

huggingface_secret = Secret.from_name("adithya-hf-wandb")
```

### Constants

```python
HOURS = 60 * 60
GPU_TYPE = "a100-80gb"  # Options: a100-80gb, a100-40gb, l40s
WANDB_PROJECT_DEFAULT = "Llama-70b-MultiGPU-finetune"
```

**GPU type considerations:**
- **A100-80GB:** Best for 70B models, expensive
- **A100-40GB:** Good for 8B-13B models, moderate cost
- **L40S:** Budget option for smaller models

---

## Building the Axolotl Image

### CUDA Base Configuration

```python
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"
```

**Why this matters:**
- Axolotl requires CUDA for compilation
- Flash Attention needs `devel` flavor
- Version 12.8.1 ensures compatibility with latest PyTorch

### Complete Image Definition

```python
AXOLOTL_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(["torch", "torchvision", "torchaudio"])
    .run_commands(
        "uv pip install --no-deps -U packaging setuptools wheel ninja --system"
    )
    .run_commands("uv pip install --no-build-isolation axolotl[deepspeed] --system")
    .run_commands(
        "UV_NO_BUILD_ISOLATION=1 uv pip install flash-attn --no-build-isolation --system"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/data/.cache",
    })
)
```

**Installation order matters:**
1. **Base dependencies** (`packaging`, `setuptools`, `wheel`, `ninja`)
2. **Axolotl with DeepSpeed** (includes distributed training support)
3. **Flash Attention** (requires `--no-build-isolation`)

> **⚠️ Build Time:** First build takes 20-30 minutes due to Flash Attention compilation. Modal caches the image—subsequent runs are instant!

### Alternative: Official Axolotl Image (Commented Out)

```python
# AXOLOTL_IMAGE = ModalImage.from_registry(
#     "axolotlai/axolotl-cloud:main-latest", add_python="3.12"
# ).env({
#     "JUPYTER_ENABLE_LAB": "no",
#     "JUPYTER_TOKEN": "",
#     "HF_HOME": "/data/.cache",
# })
```

**Why custom image?**
- Official image auto-starts JupyterLab (not needed for Modal)
- Custom image is lighter and faster
- More control over versions

---

## Training Configuration with YAML

Axolotl uses YAML for all configuration. This ensures reproducibility and easy experimentation.

### Complete Training Config

```yaml
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
wandb_project: Llama-70b-MultiGPU-finetune
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
```

### Key Configuration Sections

#### Model Settings

```yaml
base_model: NousResearch/Meta-Llama-3-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
```

**For Llama 70B:**
```yaml
base_model: meta-llama/Meta-Llama-3-70B-Instruct
```

**For other models:**
```yaml
base_model: mistralai/Mistral-7B-Instruct-v0.2
model_type: MistralForCausalLM
```

#### Quantization

```yaml
load_in_8bit: true   # Reduces memory by ~50%
load_in_4bit: false  # Reduces memory by ~75%, slight quality loss
```

**Recommendations:**
- **8B models:** `load_in_8bit: false` (full precision)
- **70B models:** `load_in_8bit: true` (essential)
- **405B models:** `load_in_4bit: true` (required)

#### Dataset Configuration

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

**Custom dataset format:**
```yaml
datasets:
  - path: your_username/your_dataset
    type: sharegpt  # or alpaca, chat_template, completion
```

#### LoRA Parameters

```yaml
adapter: lora
lora_r: 32           # Rank (higher = more capacity)
lora_alpha: 16       # Scaling factor
lora_dropout: 0.05   # Regularization
lora_target_linear: true  # Apply to all linear layers
```

**For larger models (70B+):**
```yaml
lora_r: 64
lora_alpha: 128
```

**For faster training:**
```yaml
lora_r: 16
lora_alpha: 32
```

#### Training Hyperparameters

```yaml
micro_batch_size: 8              # Batch per GPU
gradient_accumulation_steps: 4   # Effective batch = 8 × 4 = 32
num_epochs: 4
learning_rate: 0.0002
optimizer: adamw_bnb_8bit        # Memory-efficient optimizer
lr_scheduler: cosine
```

**Effective batch size calculation:**
```
Effective Batch = micro_batch_size × gradient_accumulation_steps × num_gpus
```

For 4 GPUs: `8 × 4 × 4 = 128`

#### Memory Optimizations

```yaml
gradient_checkpointing: true  # Trade compute for memory
flash_attention: true         # Faster, more memory-efficient attention
bf16: auto                    # Use bfloat16 if supported
```

**Critical for large models!**

---

## The Three-Stage Pipeline

### Helper Function: Write Config to Volume

```python
def write_config_to_volume(
    train_config_yaml: str,
    config_path: str = "/data/config.yml",
    update_paths: bool = True,
) -> dict:
    """Write YAML configuration to volume with optional path updates."""
    import os
    import yaml

    config_dict = yaml.safe_load(train_config_yaml)

    # Update paths to use volume
    if update_paths and "output_dir" in config_dict:
        config_dict["output_dir"] = config_dict["output_dir"].replace(
            "./outputs", "/data/outputs"
        )

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    exp_volume.commit()
    return config_dict
```

This helper ensures configurations are saved to the volume and paths are corrected.

---

## Stage 1: Dataset Preprocessing

```python
PREPROCESS_NUM_GPUS = 1
PREPROCESS_GPU_CONFIG = f"{GPU_TYPE}:{PREPROCESS_NUM_GPUS}"

@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
    gpu=PREPROCESS_GPU_CONFIG,
)
def process_datasets(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
):
    """Preprocess and tokenize dataset before training using Axolotl."""
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    config_dict = write_config_to_volume(train_config_yaml, config_path, True)
    exp_volume.commit()

    print("Starting dataset preprocessing...")
    try:
        subprocess.run(["axolotl", "preprocess", config_path], check=True)
        print("✓ Preprocessing completed")
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

**What happens:**
1. Writes YAML config to `/data/config.yml`
2. Runs `axolotl preprocess` to tokenize dataset
3. Saves preprocessed data to `/data/prepared_datasets/`
4. Commits to volume for training stage

**Why preprocess separately?**
- Preprocessing can take hours for large datasets
- Only needs 1 GPU (or even CPU)
- Reusable for multiple training runs
- Easier to debug

**Run it:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::process_datasets
```

**With custom config:**

```python
# Modify TRAIN_CONFIG_YAML in the script, then:
modal run FinetuneLlamaAxolotlGPUModal.py::process_datasets
```

---

## Stage 2: Multi-GPU Training

This is where the magic happens!

### GPU Configuration

```python
TRAIN_NUM_GPUS = 4  # Can be 2, 4, or 8
TRAIN_GPU_CONFIG = f"{GPU_TYPE}:{TRAIN_NUM_GPUS}"
```

**Scaling guidelines:**

| Model | Min GPUs | Recommended GPUs | GPU Type |
|-------|----------|------------------|----------|
| Llama 3-8B | 1 | 1 | A100-40GB |
| Llama 3-13B | 1 | 2 | A100-40GB |
| Llama 3-70B | 2 | 4 | A100-80GB |
| Llama 3-405B | 8 | 8 | A100-80GB |

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

    # Run Axolotl training with accelerate for multi-GPU support
    print(f"Starting training with {TRAIN_NUM_GPUS} GPUs...")
    cmd = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes", str(TRAIN_NUM_GPUS),
        "--num_machines", "1",
        "--mixed_precision", "bf16",
        "--dynamo_backend", "no",
        "-m", "axolotl.cli.train",
        config_path,
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

### Accelerate Command Breakdown

```bash
accelerate launch \
  --multi_gpu \                     # Enable multi-GPU
  --num_processes 4 \               # Number of GPUs
  --num_machines 1 \                # Single machine (Modal handles this)
  --mixed_precision bf16 \          # Use bfloat16 for speed
  --dynamo_backend no \             # Disable torch.compile (stability)
  -m axolotl.cli.train \            # Run Axolotl training module
  /data/config.yml                  # Path to config
```

**Accelerate automatically:**
- Distributes model across GPUs
- Synchronizes gradients
- Handles communication (NCCL)
- Manages checkpointing

### Run Training

**Basic:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**With custom GPU count:**

```python
# Edit in script:
TRAIN_NUM_GPUS = 8  # For massive models

# Then run:
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**Monitor training:**
1. Click URL in Modal output for live logs
2. Check W&B: `https://wandb.ai/<username>/Llama-70b-MultiGPU-finetune`
3. Monitor GPU utilization in Modal dashboard

> **💰 Cost Alert:** 4× A100-80GB costs ~$14/hour. A 3-hour training run = $42. Use preprocessing and testing to minimize trial runs!

---

## Stage 3: Merge LoRA Adapters

After training, you have LoRA adapters. Merge them into the base model for deployment.

### Merge Configuration

```python
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

**Why only 1 GPU?**
- Merging is sequential, doesn't benefit from multiple GPUs
- Saves cost

**Run it:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::merge_lora
```

**Specify custom LoRA directory:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::merge_lora \
  --lora-model-dir="/data/outputs/lora-out"
```

**Output:**
- Merged model saved to `output_dir` from config
- Can be used for inference or pushed to Hub

---

## Stage 4: Inference

Test your fine-tuned model!

```python
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

    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    print("Starting inference...")
    print(f"Prompt: {prompt}")
    print("-" * 80)

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

        return {
            "status": "completed",
            "prompt": prompt,
            "output": result.stdout,
            "model": base_model or config_dict.get("base_model"),
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Inference failed: {e}")
    finally:
        if "prompt_file" in locals():
            os.unlink(prompt_file)
```

### Run Inference

**With LoRA adapters (before merging):**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::run_inference \
  --prompt="Explain quantum computing in simple terms." \
  --lora-model-dir="/data/outputs/lora-out"
```

**With merged model:**

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::run_inference \
  --prompt="Write a poem about AI." \
  --base-model="/data/outputs/lora-out-merged"
```

**Test multiple prompts:**

```python
# Create a Python script:
import modal

app = modal.App.lookup("Finetuned_Llama_70b_Axolotl_MultiGPU")
run_inference = modal.Function.lookup(app.name, "run_inference")

prompts = [
    "Explain machine learning.",
    "Write code to sort a list.",
    "Translate 'Hello' to French.",
]

for prompt in prompts:
    result = run_inference.remote(prompt=prompt)
    print(f"Prompt: {prompt}")
    print(f"Output: {result['output']}\n")
```

---

## Complete Workflow Example

### 1. Customize Configuration

Edit `TRAIN_CONFIG_YAML` in the script:

```python
TRAIN_CONFIG_YAML = f"""
base_model: meta-llama/Meta-Llama-3-70B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: true

chat_template: llama3
datasets:
  - path: your-username/your-dataset
    type: chat_template

dataset_prepared_path: /data/prepared_datasets/custom
output_dir: /data/outputs/llama70b-lora

sequence_len: 8192
lora_r: 64
lora_alpha: 128

gradient_accumulation_steps: 2
micro_batch_size: 4
num_epochs: 3
learning_rate: 0.0001

wandb_project: {WANDB_PROJECT_DEFAULT}
"""
```

### 2. Preprocess Dataset

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::process_datasets
```

**Expected time:** 30 min - 2 hours (depending on dataset size)

### 3. Test Training (Small Run)

Edit config to use small dataset subset, then:

```bash
# Modify num_epochs: 1 in YAML
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**Expected time:** 10-30 minutes
**Cost:** ~$3-7

### 4. Full Training

```bash
# Restore full num_epochs in YAML
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
```

**Expected time:** 2-8 hours (depends on model size, dataset, GPUs)
**Cost:** $30-150

Monitor at: `https://wandb.ai/<username>/Llama-70b-MultiGPU-finetune`

### 5. Merge LoRA

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::merge_lora
```

**Expected time:** 15-45 minutes
**Cost:** ~$1-3

### 6. Test Inference

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::run_inference \
  --prompt="Test the fine-tuned model with this prompt."
```

**Expected time:** 30 seconds - 2 minutes
**Cost:** ~$0.10

### 7. Push to Hub (Optional)

Add to config:

```yaml
hub_model_id: your-username/llama-70b-finetuned
```

Axolotl will automatically push checkpoints and final model to Hugging Face Hub during training.

---

## Advanced Configuration Options

### Multi-Dataset Training

```yaml
datasets:
  - path: dataset1/name
    type: chat_template
  - path: dataset2/name
    type: alpaca
  - path: dataset3/name
    type: sharegpt
```

### Custom Evaluation

```yaml
val_set_size: 0.1           # 10% validation
evals_per_epoch: 10         # Evaluate 10 times per epoch
eval_sample_packing: false
```

### Checkpoint Management

```yaml
saves_per_epoch: 4              # Save 4 checkpoints per epoch
save_total_limit: 10            # Keep only 10 most recent
resume_from_checkpoint: /data/outputs/lora-out/checkpoint-1000
```

### Advanced LoRA

```yaml
adapter: lora
lora_r: 128
lora_alpha: 256
lora_dropout: 0.1
lora_target_modules:          # Specify exact modules
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

### Optimizer Tuning

```yaml
optimizer: adamw_bnb_8bit      # Options: adamw_torch, adamw_bnb_8bit, adafactor
lr_scheduler: cosine           # Options: linear, cosine, constant
warmup_ratio: 0.1
weight_decay: 0.01
max_grad_norm: 1.0
```

### DeepSpeed Configuration

For maximum efficiency on 8+ GPUs:

```yaml
deepspeed: /path/to/deepspeed_config.json
```

Create `deepspeed_config.json`:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    }
  },
  "fp16": {
    "enabled": true
  },
  "gradient_accumulation_steps": 4
}
```

---

## Hyperparameter Tuning Guide

### For Llama 3-8B

```yaml
micro_batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 0.0003
lora_r: 32
lora_alpha: 64
num_epochs: 3
```

**GPU:** 1× A100-40GB or 2× L40S

### For Llama 3-70B

```yaml
micro_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 0.0001
lora_r: 64
lora_alpha: 128
num_epochs: 2-3
load_in_8bit: true
```

**GPU:** 4× A100-80GB

### For Maximum Speed

```yaml
flash_attention: true
bf16: auto
gradient_checkpointing: true
sample_packing: true              # Pack multiple samples per sequence
pad_to_sequence_len: true
```

### For Maximum Quality

```yaml
learning_rate: 0.00005            # Lower LR
num_epochs: 5                     # More epochs
warmup_ratio: 0.2                 # Longer warmup
lora_r: 128                       # Higher capacity
lora_alpha: 256
```

---

## Common Issues and Solutions

### Issue 1: "Out of Memory" During Training

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   micro_batch_size: 2
   gradient_accumulation_steps: 8
   ```

2. **Enable gradient checkpointing:**
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

### Issue 2: Training Loss Not Decreasing

**Solutions:**

1. **Increase learning rate:**
   ```yaml
   learning_rate: 0.0003
   ```

2. **Check data quality:**
   - Verify dataset format
   - Check for corrupted samples

3. **Increase LoRA rank:**
   ```yaml
   lora_r: 64
   lora_alpha: 128
   ```

4. **Train longer:**
   ```yaml
   num_epochs: 5
   ```

### Issue 3: Preprocessing Hangs

**Solution:**

Check dataset path and HF token:

```bash
# Test dataset access locally
from datasets import load_dataset
dataset = load_dataset("your/dataset", split="train")
print(dataset)
```

If it works locally, ensure HF token is in Modal secrets.

### Issue 4: Multi-GPU Not Working

**Symptoms:** Training only uses 1 GPU

**Solutions:**

1. **Verify GPU count:**
   ```python
   TRAIN_NUM_GPUS = 4  # Must match your intent
   ```

2. **Check accelerate config:**
   ```bash
   # In training function, before training:
   os.system("accelerate env")  # Prints config
   ```

3. **Ensure NCCL is available:**
   ```python
   import torch
   print(torch.cuda.nccl.version())
   ```

### Issue 5: "Secret not found"

**Solution:**

```bash
# Create the secret
modal secret create adithya-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxx \
  WANDB_API_KEY=xxx

# Or update script to use different secret name
huggingface_secret = Secret.from_name("my-secret-name")
```

---

## Cost Optimization Strategies

### 1. Preprocessing Separately

**Bad (expensive):**
```bash
# Preprocess + train in one run on 4 A100s
modal run script.py::train_model  # Preprocessing on 4 GPUs!
```

**Good (cheap):**
```bash
# Preprocess on 1 GPU
modal run script.py::process_datasets  # $3.50/hr × 1 GPU

# Train on 4 GPUs
modal run script.py::train_model  # $14/hr × 4 GPUs
```

**Savings:** ~$10/hr during preprocessing

### 2. Use Smaller GPUs for Testing

```python
# Testing
GPU_TYPE = "l40s"  # ~$1/hr

# Production
GPU_TYPE = "a100-80gb"  # ~$3.50/hr
```

### 3. Limit Test Datasets

```yaml
# In YAML config for testing:
datasets:
  - path: dataset/name
    type: chat_template
    # Add this to limit samples:
    num_samples: 100
```

### 4. Checkpoint Smartly

```yaml
saves_per_epoch: 2           # Don't save too frequently
save_total_limit: 5          # Delete old checkpoints
```

**Volume storage is free up to 50GB**, but good practice.

### 5. Use Resume from Checkpoint

If training fails midway:

```yaml
resume_from_checkpoint: /data/outputs/lora-out/checkpoint-500
```

**Don't start from scratch!**

---

## Monitoring and Debugging

### View Real-time Logs

```bash
modal run FinetuneLlamaAxolotlGPUModal.py::train_model
# Click the URL to open Modal dashboard
```

**Dashboard shows:**
- Real-time logs
- GPU utilization per GPU
- Memory usage
- Cost accumulation

### Check GPU Utilization

All GPUs should be near 100% during training. If not:

1. **Batch size too small:** Increase `micro_batch_size`
2. **Data loading bottleneck:** Enable `sample_packing`
3. **CPU preprocessing slow:** Use more CPU cores (Modal auto-optimizes)

### Debugging Training Issues

**Enable verbose logging:**

```yaml
logging_steps: 1     # Log every step
```

**Check specific checkpoint:**

```bash
modal volume ls Finetuned_Llama_70b_Axolotl /data/outputs/lora-out
# Lists all checkpoints
```

**Download checkpoint locally:**

```bash
modal volume get Finetuned_Llama_70b_Axolotl \
  /data/outputs/lora-out/checkpoint-1000 \
  ./local_checkpoint
```

---

## Scaling to 8 GPUs

For massive models (Llama 405B, Mixtral 8×22B):

### Update Configuration

```python
TRAIN_NUM_GPUS = 8
GPU_TYPE = "a100-80gb"
```

### Update YAML

```yaml
micro_batch_size: 2
gradient_accumulation_steps: 2
# Effective batch: 2 × 2 × 8 = 32
```

### Enable DeepSpeed ZeRO Stage 3

Create `deepspeed_zero3.json`:

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

Update YAML:

```yaml
deepspeed: /path/to/deepspeed_zero3.json
```

**Cost:** 8× A100-80GB = ~$28/hour

---

## Next Steps

- **Custom datasets:** Format your data for Axolotl
- **Advanced LoRA:** Experiment with QLoRA, DoRA
- **Evaluation:** Add custom eval metrics
- **Deployment:** Serve with vLLM (see Gemma tutorial)
- **Multi-node training:** Scale beyond 8 GPUs (advanced)

---

## Resources

- [Axolotl Documentation](https://docs.axolotl.ai/)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples)
- [Modal Documentation](https://modal.com/docs)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

---

## Complete Script Reference

The full script includes:

- **Lines 1-77:** Configuration, image setup
- **Lines 83-106:** Helper function for config management
- **Lines 114-170:** Training configuration (YAML)
- **Lines 181-215:** Dataset preprocessing function
- **Lines 226-301:** Multi-GPU training function
- **Lines 312-372:** LoRA merging function
- **Lines 383-481:** Inference function

All functions are modular and can run independently!
