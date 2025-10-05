# Fine-tuning Gemma 3-4B with Unsloth on Modal: End-to-End Vision-Language Training

This comprehensive tutorial covers fine-tuning Google's Gemma 3-4B vision-language model using Unsloth on Modal's serverless infrastructure. We'll train on LaTeX OCR dataset, evaluate the model, and deploy it for inference using vLLM.

## Why Unsloth?

**Unsloth** is an optimized library for efficient single-GPU fine-tuning with:
- 2x faster training than standard methods
- 60-80% less memory usage
- Built-in support for LoRA and QLoRA
- Simple API for vision-language models

**Modal** provides the serverless infrastructure to run training, evaluation, and serving without managing GPUs.

## What You'll Build

A complete ML pipeline with:
1. **Dataset download and caching**
2. **Model download and preparation**
3. **LoRA fine-tuning** on vision-language tasks
4. **Model evaluation** with metrics
5. **Production serving** with vLLM

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
- **Hugging Face token** (for model/dataset access)
- **Weights & Biases API key** (optional, for experiment tracking)

#### Option 1: Use .env file

Create `.env` in your project directory:

```bash
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxx
```

#### Option 2: Create Modal Secret

```bash
modal secret create adithya-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx \
  WANDB_API_KEY=xxxxxxxxxxxxx
```

> **Note:** The script expects a secret named `adithya-hf-wandb`. Either create one with this name or modify `huggingface_secret = Secret.from_name("adithya-hf-wandb")` in the code.

---

## Project Structure

```
ServerLessFinetuning/
├── FinetuneGemmaUnslothModal.py    # Complete training pipeline
└── .env                             # Your secrets (optional)
```

No repository cloning needed—everything runs from a single Python file!

---

## Understanding the Complete Pipeline

This script demonstrates a **production-grade** ML workflow with 6 stages:

1. **Dataset Download**
2. **Model Download**
3. **Fine-tuning with LoRA**
4. **Model Export/Merge**
5. **vLLM Serving**
6. **Evaluation**

### Architecture Overview

```
┌─────────────────┐
│  Download Data  │  (CPU)
└────────┬────────┘
         │
┌────────▼────────┐
│ Download Model  │  (L40S GPU)
└────────┬────────┘
         │
┌────────▼────────┐
│   Fine-tune     │  (A100-80GB GPU)
│   with LoRA     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Export/Merge   │  (A100-80GB GPU)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│ Serve │ │ Evaluate│
│ vLLM  │ │  Model  │
└───────┘ └─────────┘
```

---

## Configuration and Setup

### App, Volume, and Secret Setup

```python
from modal import App, Image as ModalImage, Volume, Secret

app = App("Finetuned_Gemma_3_4b_it")

# Persistent storage for models, datasets, checkpoints
exp_volume = Volume.from_name("Finetuned_Gemma_3_4b_it", create_if_missing=True)
VOLUME_CONFIG = {
    "/data": exp_volume,
}

# Secret for HuggingFace and W&B access
huggingface_secret = Secret.from_name("adithya-hf-wandb")
```

**What's happening:**
- `Volume`: Stores datasets, models, checkpoints persistently
- All functions mount this volume at `/data`
- Secrets inject environment variables securely

> **💡 Volume Strategy:** Using a single volume for the entire experiment keeps everything organized. All artifacts are in `/data`.

### Configuration Constants

```python
HOURS = 60 * 60
BASE_MODEL_NAME = "unsloth/gemma-3-4b-it"
WANDB_PROJECT_DEFAULT = "GemmaFinetuning"
OUTPUT_DIR_DEFAULT = "/data/Finetuned_Gemma_3_4b_it"
```

These constants make it easy to:
- Change the base model
- Organize outputs
- Configure W&B tracking

---

## Building the Training Image

This is crucial—the image determines what software is available.

### CUDA Base Image

```python
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"
```

**Why CUDA devel?**
- `devel` includes CUDA compiler (`nvcc`) needed for building extensions
- Required for packages like `flash-attn` and `triton`
- `runtime` images won't work for compilation

### Complete Image Definition

```python
FINETUNING_GPU_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(["torch", "torchvision", "torchaudio"])
    .uv_pip_install([
        # Unsloth core
        "unsloth",
        "unsloth_zoo",
        # Quantization and efficiency
        "bitsandbytes",
        "accelerate",
        "xformers",
        "peft",
        "trl",
        "triton",
        "cut_cross_entropy",
        # Transformers ecosystem
        "transformers",
        "timm",
        # Training tools
        "wandb",
        "weave",
        "deepspeed",
        # Evaluation metrics
        "nltk",
        "rouge_score",
        "bert_score",
        "jiwer",
        "scikit-learn",
        # Utilities
        "pillow",
        "opencv-python-headless",
        "gradio",
        "hf_transfer",
    ])
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads
        "HF_HOME": "/data/.cache",         # Cache under volume
    })
)
```

**Key points:**
- **`uv_pip_install`**: Fast, modern pip alternative
- **`HF_HUB_ENABLE_HF_TRANSFER`**: Enables parallel downloads (much faster!)
- **`HF_HOME`**: Caches models/datasets in volume (persists across runs)

> **⚠️ Image Build Time:** First build takes 10-15 minutes. Modal caches the image, so subsequent runs are instant!

---

## Stage 1: Download Datasets

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
)
def download_datasets(
    dataset_name: str = "unsloth/LaTeX_OCR",
    split: str = "train",
    cache_dir: str = "/data/.cache",
):
    """Download and cache a dataset from Hugging Face."""
    from datasets import load_dataset
    import os

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    print("\n✓ Dataset loaded successfully!")
    print(f"  - Name: {dataset_name}")
    print(f"  - Split: {split}")
    print(f"  - Number of samples: {len(dataset)}")

    exp_volume.commit()  # Persist to volume!

    return {
        "status": "completed",
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
    }
```

**Why separate dataset download?**
- No GPU needed (saves money)
- Can run during development/testing
- Cached for all subsequent functions

**Run it:**

```bash
modal run FinetuneGemmaUnslothModal.py::download_datasets
```

**Custom dataset:**

```bash
modal run FinetuneGemmaUnslothModal.py::download_datasets \
  --dataset-name="your/dataset" \
  --split="train"
```

---

## Stage 2: Download Models

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    gpu="l40s:1",
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
)
def download_models(
    model_name: str = BASE_MODEL_NAME,
    cache_dir: str = "/data/.cache",
):
    """Download and cache a model using FastVisionModel."""
    from unsloth import FastVisionModel
    import os
    import torch

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    model, processor = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=8000,
        dtype=torch.bfloat16,
    )

    exp_volume.commit()

    return {
        "status": "completed",
        "model_name": model_name,
        "cache_dir": cache_dir,
    }
```

**Why L40S GPU for download?**
- Some models require GPU for initialization
- L40S is cheaper than A100
- Just for downloading, not training

**Run it:**

```bash
modal run FinetuneGemmaUnslothModal.py::download_models
```

> **💾 Caching Magic:** Model weights download to `/data/.cache` (on volume). Training functions reuse this cache—no re-downloading!

---

## Stage 3: Fine-Tuning with LoRA

This is the core training function. Let's break it down:

### GPU Configuration

```python
TRAIN_GPU = "a100-80gb"
NUM_GPUS = 1
TRAINING_GPU_CONFIG = f"{TRAIN_GPU}:{NUM_GPUS}"
```

**GPU choices:**
- **L40S**: Budget option, slower but cheaper
- **A100-40GB**: Good for smaller models
- **A100-80GB**: Best for Gemma 3-4B with vision

### Function Signature

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, Secret.from_dotenv()],
    gpu=TRAINING_GPU_CONFIG,
    timeout=24 * HOURS,
)
def fine_tune_unsloth(
    model_path: str = BASE_MODEL_NAME,
    dataset_name: str = "unsloth/LaTeX_OCR",
    dataset_split: str = "train",
    output_dir: str = OUTPUT_DIR_DEFAULT,
    hub_id: str = None,  # Push to HF Hub
    max_samples: int = None,  # Limit dataset size
    # LoRA parameters
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    # Training hyperparameters
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    learning_rate: float = 3e-4,
    warmup_ratio: float = 0.2,
    max_seq_length: int = 8000,
    # Checkpointing
    save_strategy: str = "steps",
    save_steps: int = 250,
    save_total_limit: int = 20,
    logging_steps: int = 10,
    # WandB
    wandb_project: str = WANDB_PROJECT_DEFAULT,
    wandb_run_name: str = None,
):
```

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 32 | LoRA rank (higher = more capacity, slower) |
| `lora_alpha` | 64 | LoRA scaling factor |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch = 4 × 4 = 16 |
| `max_samples` | None | Limit training samples (for testing) |
| `hub_id` | None | Push to HF Hub (e.g., "username/model-name") |

### Environment Setup

```python
os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
os.environ["WANDB_PROJECT"] = wandb_project

# Auto-generate run name
if wandb_run_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_path.split("/")[-1]
    wandb_run_name = f"finetune_{model_short}_{timestamp}"

os.environ["WANDB_RUN_NAME"] = wandb_run_name

# Memory optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch._dynamo.config.disable = True
```

> **🔍 W&B Integration:** Automatically logs metrics, gradients, system stats. Check `wandb.ai/<your-project>` during training.

### Load Model with LoRA

```python
# Load base model
model, processor = FastVisionModel.from_pretrained(
    model_path,
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
)

# Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,    # Don't train vision encoder
    finetune_language_layers=True,   # Train language model
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    random_state=3407,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],
)

# Set chat template
processor = get_chat_template(processor, "gemma-3")
```

**LoRA strategy:**
- Vision encoder frozen (pre-trained features are good!)
- Language model adapters added
- Only ~1-2% of parameters trained
- Massive memory savings

### Dataset Preprocessing

```python
dataset = load_dataset(dataset_name, split=dataset_split)

# Limit dataset for testing
if max_samples is not None and max_samples > 0:
    dataset = dataset.select(range(min(max_samples, len(dataset))))

instruction = "Write the LaTeX representation for this image."

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["text"]}],
        },
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
```

**Format:** Chat-style messages with image + text input, text output.

### Training with SFTTrainer

```python
FastVisionModel.for_training(model)  # Enable training mode

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model=model, processor=processor),
    args=SFTConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        # Optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        optim="adamw_torch_fused",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir=output_dir,
        report_to="wandb",
        # Vision-specific
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=max_seq_length,
    ),
)

# Train!
trainer_stats = trainer.train()
```

**Optimizer:** `adamw_torch_fused` is PyTorch's fastest AdamW implementation.

### Save and Push Models

```python
final_weights_dir = os.path.join(output_dir, "final_weights")
final_lora_dir = os.path.join(output_dir, "final_lora")

# Save LoRA adapters only
model.save_pretrained(final_lora_dir)
processor.save_pretrained(final_weights_dir)

if hub_id:
    # Push LoRA to Hub
    model.push_to_hub(f"{hub_id}_lora", token=os.environ["HUGGINGFACE_TOKEN"])
    processor.push_to_hub(f"{hub_id}_lora", token=os.environ["HUGGINGFACE_TOKEN"])

# Save merged model (base + LoRA)
model.save_pretrained_merged(
    final_weights_dir, processor, save_method="merged_16bit"
)

if hub_id:
    # Push merged model to Hub
    model.push_to_hub_merged(
        hub_id,
        processor,
        token=os.environ["HUGGINGFACE_TOKEN"],
        save_method="merged_16bit",
    )

exp_volume.commit()  # Persist everything!
```

**Two save formats:**
1. **LoRA adapters** (`final_lora/`): Small (~100MB), requires base model
2. **Merged model** (`final_weights/`): Full model, ready to deploy

### Running Training

**Basic training:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth
```

**Test on small subset:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --max-samples=100 \
  --num-train-epochs=1
```

**Full training with Hub push:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --hub-id="your-username/gemma-latex-ocr" \
  --num-train-epochs=3 \
  --learning-rate=0.0003
```

**Custom hyperparameters:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --lora-r=64 \
  --lora-alpha=128 \
  --per-device-train-batch-size=2 \
  --gradient-accumulation-steps=8
```

---

## Stage 4: Export and Merge Model

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    gpu=TRAINING_GPU_CONFIG,
    secrets=[huggingface_secret, Secret.from_dotenv()],
    timeout=2 * HOURS,
)
def export_model(
    lora_model_path: str = f"{OUTPUT_DIR_DEFAULT}",
    output_path: str = None,
    hub_model_id: str = None,
    push_to_hub: bool = True,
):
    """Export and merge LoRA weights with base model."""
    from unsloth import FastVisionModel
    import os

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Load LoRA model
    model, processor = FastVisionModel.from_pretrained(
        lora_model_path,
        load_in_4bit=False,
    )

    # Prepare for inference (merges weights)
    FastVisionModel.for_inference(model)

    if push_to_hub and hub_model_id:
        model.push_to_hub_merged(
            hub_model_id,
            processor,
            token=os.environ["HUGGINGFACE_TOKEN"],
            save_method="merged_16bit",
        )
    else:
        if output_path is None:
            output_path = f"{lora_model_path}_merged"
        model.save_pretrained_merged(output_path, processor, save_method="merged_16bit")

    exp_volume.commit()
```

**When to use:**
- If you only saved LoRA adapters during training
- Want to create a standalone merged model later

**Run it:**

```bash
modal run FinetuneGemmaUnslothModal.py::export_model \
  --lora-model-path="/data/Finetuned_Gemma_3_4b_it/final_lora" \
  --hub-model-id="username/gemma-latex-merged"
```

---

## Stage 5: Serving with vLLM

Deploy your fine-tuned model for high-throughput inference!

### vLLM Image

```python
VLLM_CUDA_VERSION = "12.8.1"
VLLM_CUDA_TAG = f"{VLLM_CUDA_VERSION}-devel-ubuntu24.04"

VLLM_GPU_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{VLLM_CUDA_TAG}", add_python="3.12")
    .apt_install("libopenmpi-dev", "libnuma-dev")
    .run_commands("pip install --upgrade pip")
    .run_commands("pip install uv")
    .run_commands("uv pip install vllm -U --system")
    .pip_install(
        "datasets",
        "pillow",
        "huggingface_hub[hf_transfer]",
        "requests",
        "numpy",
    )
    .run_commands(
        "uv pip install 'flash-attn>=2.7.1,<=2.8.0' --no-build-isolation --system"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

**Why separate image?**
- vLLM has different dependencies than training
- Optimized for serving, not training
- Smaller image size

### Serving Configuration

```python
DEFAULT_SERVE_MODEL = "/data/Finetuned_Gemma_3_4b_it/final_weights"
SERVE_GPU = "L40S"
SERVE_NUM_GPUS = 1
VLLM_PORT = 8000
```

### Serve Function

```python
@app.function(
    image=VLLM_GPU_IMAGE,
    gpu=SERVE_GPU_CONFIG,
    scaledown_window=3 * 60,  # Scale down after 3 min idle
    secrets=[huggingface_secret],
    volumes=VOLUME_CONFIG,
    max_containers=2,  # Auto-scale up to 2 containers
    timeout=24 * HOURS,
)
@modal.concurrent(max_inputs=50)  # Handle 50 concurrent requests
@modal.web_server(port=8000, startup_timeout=5 * 60)
def serve_vllm():
    """Serve model using vLLM."""
    import subprocess
    import os

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=info",
        DEFAULT_SERVE_MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--enforce-eager",  # Faster startup
        "--tensor-parallel-size", str(SERVE_NUM_GPUS),
        "--gpu-memory-utilization", "0.4",
        "--trust-remote-code",
    ]

    subprocess.Popen(" ".join(cmd), shell=True)
```

**Key features:**
- **Auto-scaling:** Scales from 0 to 2 containers based on load
- **Scale-down:** Shuts down after 3 minutes of inactivity (saves cost!)
- **Concurrent:** Handles 50 requests per container
- **OpenAI-compatible API:** Use with OpenAI client

### Deploy the Server

```bash
modal deploy FinetuneGemmaUnslothModal.py
```

This creates a persistent endpoint (stays alive across Modal sessions).

**Get the URL:**

```bash
modal app list  # Find your app
# Output: https://your-username--finetuned-gemma-3-4b-it-serve-vllm.modal.run
```

### Test the API

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="https://your-endpoint.modal.run/v1",
    api_key="EMPTY"
)

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="/data/Finetuned_Gemma_3_4b_it/final_weights",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "Write the LaTeX representation for this image."},
            ],
        },
    ],
    temperature=0.1,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

> **💰 Cost Optimization:** The `scaledown_window=3*60` setting means the server shuts down automatically after 3 minutes with no requests. You only pay for active time!

---

## Stage 6: Evaluation

Measure your model's performance with automated metrics.

### Evaluation Image

```python
EVAL_IMAGE = (
    ModalImage.debian_slim(python_version="3.12")
    .pip_install(
        "openai",
        "datasets",
        "pillow",
        "numpy",
        "jiwer",  # Word/Character Error Rate
        "nltk",
        "tqdm",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

**No GPU needed!** Evaluation runs on CPU, calls your vLLM endpoint.

### Evaluation Function

```python
@app.function(
    image=EVAL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=2 * HOURS,
)
def evaluate_model(
    endpoint_url: str = None,  # Auto-detects if None
    model_name: str = "/data/Finetuned_Gemma_3_4b_it/final_weights",
    dataset_name: str = "unsloth/LaTeX_OCR",
    dataset_split: str = "test",
    max_samples: int = 100,
    max_parallel_requests: int = 8,
    temperature: float = 0.1,
    max_tokens: int = 512,
):
    """Evaluate model on LaTeX OCR dataset."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from datasets import load_dataset
    from jiwer import wer, cer
    from tqdm import tqdm

    # Auto-retrieve endpoint URL
    if endpoint_url is None:
        endpoint_url = serve_vllm.get_web_url().rstrip("/") + "/v1"

    # Load test dataset
    dataset = load_dataset(dataset_name, split=dataset_split)
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    client = OpenAI(base_url=endpoint_url, api_key="EMPTY")

    # Parallel inference
    def run_inference(sample, idx):
        image_base64 = encode_image_to_base64(sample["image"])
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": "Write the LaTeX representation for this image."},
                    ],
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "prediction": response.choices[0].message.content.strip(),
            "ground_truth": sample["text"].strip(),
        }

    # Run evaluation
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        futures = [executor.submit(run_inference, dataset[i], i) for i in range(len(dataset))]
        results = [f.result() for f in tqdm(as_completed(futures), total=len(dataset))]

    # Calculate metrics
    predictions = [r["prediction"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]

    word_error_rate = wer(ground_truths, predictions)
    char_error_rate = cer(ground_truths, predictions)
    exact_match_accuracy = sum(p == g for p, g in zip(predictions, ground_truths)) / len(results)

    return {
        "metrics": {
            "exact_match_accuracy": exact_match_accuracy,
            "character_error_rate": char_error_rate,
            "word_error_rate": word_error_rate,
        },
        "examples": results[:10],  # First 10 examples
    }
```

**Metrics:**
- **Exact Match**: Percentage of perfect predictions
- **Character Error Rate (CER)**: Edit distance at character level
- **Word Error Rate (WER)**: Edit distance at word level

### Run Evaluation

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model
```

**Custom parameters:**

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model \
  --max-samples=500 \
  --max-parallel-requests=16 \
  --temperature=0.0
```

**Specify endpoint manually:**

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model \
  --endpoint-url="https://your-endpoint.modal.run/v1"
```

---

## Complete Workflow Example

### 1. Download Everything

```bash
# Download dataset (runs on CPU)
modal run FinetuneGemmaUnslothModal.py::download_datasets

# Download model (runs on L40S)
modal run FinetuneGemmaUnslothModal.py::download_models
```

### 2. Quick Test Training

```bash
# Train on 100 samples to verify everything works
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --max-samples=100 \
  --num-train-epochs=1 \
  --save-steps=50
```

### 3. Full Training Run

```bash
# Production training with Hub push
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --hub-id="username/gemma-latex-ocr" \
  --num-train-epochs=3 \
  --learning-rate=0.0003 \
  --per-device-train-batch-size=4 \
  --gradient-accumulation-steps=4
```

Monitor on W&B: `https://wandb.ai/<username>/GemmaFinetuning`

### 4. Deploy Serving Endpoint

```bash
modal deploy FinetuneGemmaUnslothModal.py
```

Get URL from output or:

```bash
modal app list
```

### 5. Evaluate

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model \
  --max-samples=500
```

---

## Hyperparameter Tuning Guide

### LoRA Parameters

**Higher capacity (slower, more memory):**
```bash
--lora-r=64 --lora-alpha=128
```

**Lower capacity (faster, less memory):**
```bash
--lora-r=16 --lora-alpha=32
```

**Rule of thumb:** `lora_alpha = 2 * lora_r`

### Batch Size and Gradient Accumulation

Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

**For A100-80GB:**
```bash
--per-device-train-batch-size=8 --gradient-accumulation-steps=2  # Effective: 16
```

**If OOM (Out of Memory):**
```bash
--per-device-train-batch-size=2 --gradient-accumulation-steps=8  # Still effective: 16
```

### Learning Rate

**Conservative (safe):**
```bash
--learning-rate=0.0001
```

**Aggressive (faster convergence, risky):**
```bash
--learning-rate=0.0005
```

**Typical range:** `1e-4` to `5e-4`

### Sequence Length

**Shorter (faster, less memory):**
```bash
--max-seq-length=4096
```

**Longer (better for long documents):**
```bash
--max-seq-length=16384
```

---

## Common Issues and Solutions

### Issue 1: "Secret not found"

**Error:** `Secret "adithya-hf-wandb" not found`

**Solution:**

```bash
modal secret create adithya-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxx \
  WANDB_API_KEY=xxx
```

Or use `.env` file and change:
```python
secrets=[Secret.from_dotenv()]
```

### Issue 2: CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**

1. Reduce batch size:
   ```bash
   --per-device-train-batch-size=2
   ```

2. Reduce sequence length:
   ```bash
   --max-seq-length=4096
   ```

3. Use smaller LoRA rank:
   ```bash
   --lora-r=16 --lora-alpha=32
   ```

4. Switch to larger GPU:
   ```python
   TRAIN_GPU = "a100-80gb"
   ```

### Issue 3: Image Build Timeout

**Error:** Image build exceeds timeout

**Solution:** Image builds can take 15-20 minutes first time. This is normal. Subsequent builds use cache and are instant.

If it truly times out, simplify the image by removing non-essential packages.

### Issue 4: Training Too Slow

**Solutions:**

1. Use larger GPU:
   ```python
   TRAIN_GPU = "a100-80gb"
   ```

2. Increase batch size (if memory allows):
   ```bash
   --per-device-train-batch-size=8
   ```

3. Reduce dataset:
   ```bash
   --max-samples=5000
   ```

### Issue 5: Evaluation Fails

**Error:** `Could not connect to endpoint`

**Solution:** Make sure serving endpoint is deployed:

```bash
modal app list  # Check if serve_vllm is running
```

If not running:

```bash
modal deploy FinetuneGemmaUnslothModal.py
```

---

## Cost Estimation

Based on Modal pricing (approximate):

### Training
- **A100-80GB:** ~$3.50/hour
- **Full training run (3 epochs, 10K samples):** ~2-3 hours = **$7-10**

### Serving (Pay per use)
- **L40S:** ~$1/hour
- **Auto-scales to zero** after 3 minutes idle
- **Monthly cost with moderate usage:** $5-20

### Storage
- **Volumes:** Free up to 50GB
- **Model + dataset + checkpoints:** ~15GB = **$0/month**

### Total for Complete Pipeline
- **One-time training:** $10-15
- **Monthly serving:** $5-20 (depending on usage)

> **💡 Pro Tip:** Delete old checkpoints to save space:
> ```bash
> modal volume ls nanogpt-outputs  # List files
> modal volume rm nanogpt-outputs /path/to/old/checkpoint  # Delete
> ```

---

## Advanced: Multi-GPU Training

For even faster training, use multiple GPUs:

```python
# Change in the script
NUM_GPUS = 2
TRAIN_GPU_CONFIG = f"a100-80gb:{NUM_GPUS}"
```

Unsloth doesn't natively support multi-GPU. For multi-GPU, use the Axolotl example instead (see `FinetuneLlamaAxolotlGPUModal.md`).

---

## Monitoring and Debugging

### View Logs in Real-time

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth
# Click the URL in output to view dashboard
```

### Check Volume Contents

```bash
modal volume ls Finetuned_Gemma_3_4b_it
```

### Download from Volume

```bash
modal volume get Finetuned_Gemma_3_4b_it /data/Finetuned_Gemma_3_4b_it/final_weights ./local_model
```

### Check GPU Utilization

In the Modal dashboard:
- Click on running function
- View "GPU utilization" graph
- Should be near 100% during training

---

## Next Steps

- **Try different datasets:** Replace LaTeX OCR with your custom dataset
- **Experiment with prompts:** Change instruction in `convert_to_conversation()`
- **Add validation:** Implement validation split evaluation during training
- **Deploy to production:** Use Modal's persistent deployments
- **Scale serving:** Increase `max_containers` for high-traffic scenarios

---

## Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Modal Documentation](https://modal.com/docs)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Gemma Model Card](https://huggingface.co/google/gemma-3-4b-it)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

## Complete Script Reference

The complete script is available in `FinetuneGemmaUnslothModal.py`. Key sections:

- **Lines 1-98:** App setup, configuration, image definition
- **Lines 102-150:** Dataset download function
- **Lines 152-196:** Model download function
- **Lines 204-507:** Main fine-tuning function
- **Lines 509-599:** Model export function
- **Lines 642-693:** vLLM serving function
- **Lines 716-947:** Evaluation function

Each function is self-contained and can run independently!
