import os
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# MODAL APP DEFINITION , VOLUME AND SECRET SETUP
# =============================================================================

app = App("Finetuned_Gemma_3_4b_it")

# Create volumes for persistent storage
exp_volume = Volume.from_name("Finetuned_Gemma_3_4b_it", create_if_missing=True)
# Configure volume mounting points
VOLUME_CONFIG = {
    "/data": exp_volume,
}
huggingface_secret = Secret.from_name("secrets-hf-wandb")


# =============================================================================
# CONFIGURATION DEFAULT CONSTANTS
# =============================================================================

# Time constants
HOURS = 60 * 60
# Model Configuration
BASE_MODEL_NAME = "unsloth/gemma-3-4b-it"
WANDB_PROJECT_DEFAULT = "GemmaFinetuning"
OUTPUT_DIR_DEFAULT = "/data/Finetuned_Gemma_3_4b_it"


# =============================================================================
# CONFIGURE IMAGES AND ENVIRONMENTS
# =============================================================================

# CUDA Configuration for SGLang
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Define the GPU image for fine-tuning with Unsloth
FINETUNING_GPU_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
    )
    .uv_pip_install(
        [
            "torch",
            "torchvision",
            "torchaudio",  # optional but often bundled with torch
        ]
    )
    # Install Unsloth and dependencies
    .uv_pip_install(
        [
            # Unsloth core packages
            "unsloth",
            "unsloth_zoo",
            # Core ML packages
            "bitsandbytes",
            "accelerate",
            "xformers",
            "peft",
            "trl",
            "triton",
            "cut_cross_entropy",
            # Upgraded packages
            "transformers",
            "timm",
            # Additional dependencies
            "wandb",
            "weave",
            "pillow",
            "opencv-python-headless",
            "deepspeed",
            "pyyaml",
            "packaging",
            "nltk",
            "rouge_score",
            "bert_score",
            "jiwer",
            "scikit-learn",
            "tqdm",
            "pandas",
            "pyarrow",
            "gradio",
            "hf_transfer",
        ]
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",  # Set HF cache root under /data
        }
    )
)


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
    """
    Download and cache a dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset to download (e.g., 'unsloth/LaTeX_OCR')
        split: Dataset split to download (e.g., 'train', 'test', 'validation')
        cache_dir: Directory to cache the dataset

    Returns:
        dict: Contains status, dataset info, and cache location
    """
    from datasets import load_dataset
    import os

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    print("\n✓ Dataset loaded successfully!")
    print(f"  - Name: {dataset_name}")
    print(f"  - Split: {split}")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Cached at: {cache_dir}")
    print("\nDataset structure:")
    print(dataset)

    exp_volume.commit()

    return {
        "status": "completed",
        "dataset_name": dataset_name,
        "split": split,
        "num_samples": len(dataset),
        "cache_dir": cache_dir,
    }


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
    """
    Download and cache a model from Hugging Face using FastVisionModel.
    Uses 4-bit quantization for memory efficiency.

    Args:
        model_name: Name of the model to download (e.g., 'unsloth/gemma-3-4b-it')
        cache_dir: Base directory to cache the model

    Returns:
        dict: Contains status and model info
    """
    from unsloth import FastVisionModel
    import os
    import torch

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    model, processor = FastVisionModel.from_pretrained(
        model_name,  # Can be HF hub ID or local path
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=8000,
        dtype=torch.bfloat16,  # Use bfloat16 for better performance
    )
    # Commit the volume to persist the cached model
    exp_volume.commit()

    return {
        "status": "completed",
        "model_name": model_name,
        "cache_dir": cache_dir,
        "quantization": "4-bit",
    }


# GPU Configuration
TRAIN_GPU = "a100-80gb"  # Default GPU for training
NUM_GPUS = 1
TRAINING_GPU_CONFIG = f"{TRAIN_GPU}:{NUM_GPUS}"


@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, Secret.from_dotenv()],
    gpu=TRAINING_GPU_CONFIG,
    timeout=24 * HOURS,
)
def fine_tune_unsloth(
    model_path: str = BASE_MODEL_NAME,  # Can be HF hub ID or local path
    dataset_name: str = "unsloth/LaTeX_OCR",
    dataset_split: str = "train",
    output_dir: str = OUTPUT_DIR_DEFAULT,
    hub_id: str = None,
    max_samples: int = None,  # Maximum number of samples to use from dataset
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
    # Checkpoint saving configuration
    save_strategy: str = "steps",
    save_steps: int = 250,
    save_total_limit: int = 20,
    logging_steps: int = 10,
    # WandB config
    wandb_project: str = WANDB_PROJECT_DEFAULT,
    wandb_run_name: str = None,
):
    """
    Fine-tune a vision-language model using Unsloth with LoRA.

    Args:
        model_path: Hugging Face model ID or local path to base model
        dataset_name: Name of the dataset to use for training
        dataset_split: Dataset split to use
        output_dir: Directory to save the fine-tuned model
        hub_id: Hugging Face Hub ID to push the model to (optional, if None, model won't be pushed)
        max_samples: Maximum number of samples to use from dataset (if None, use all samples)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio for learning rate scheduler
        max_seq_length: Maximum sequence length
        save_strategy: Checkpoint save strategy ('steps' or 'epoch')
        save_steps: Save checkpoint every N steps (when save_strategy='steps')
        save_total_limit: Maximum number of checkpoints to keep
        logging_steps: Log metrics every N steps
        wandb_project: Weights & Biases project name
        wandb_run_name: Weights & Biases run name

    Returns:
        dict: Contains training statistics and paths
    """
    from unsloth import FastVisionModel, get_chat_template
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    import os
    import torch
    from datetime import datetime
    from datasets import load_dataset

    print(f"\n{'=' * 80}")
    print("FINE-TUNING CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name} ({dataset_split})")
    print(f"Output: {output_dir}")
    print(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(
        f"Training: batch_size={per_device_train_batch_size}, "
        f"grad_accum={gradient_accumulation_steps}, epochs={num_train_epochs}"
    )
    print(f"{'=' * 80}\n")
    os.makedirs(output_dir, exist_ok=True)

    # Set up environment variables

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["WANDB_PROJECT"] = wandb_project

    # Create a meaningful run name if not provided
    if wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_path.split("/")[-1]  # Get just the model name part
        wandb_run_name = f"finetune_{model_short}_{timestamp}"

    # Set the W&B run name
    os.environ["WANDB_RUN_NAME"] = wandb_run_name
    print(f"W&B Run Name: {wandb_run_name}")

    # Swift-compatible memory optimization
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable dynamo for stable training
    torch._dynamo.config.disable = True

    print("Loading Unsloth model...")

    # =============================================================================
    # Load model and add LoRA adapters
    # =============================================================================

    print(f"Loading model from: {model_path}")
    model, processor = FastVisionModel.from_pretrained(
        model_path,  # Can be HF hub ID or local path
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,  # Use bfloat16 for better performance
    )

    # Add LoRA adapters
    print(
        f"Adding LoRA adapters (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})..."
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        random_state=3407,
        target_modules="all-linear",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    # Set up chat template
    processor = get_chat_template(processor, "gemma-3")

    # =============================================================================
    # Load and preprocess dataset
    # =============================================================================

    print(f"Loading dataset: {dataset_name} (split: {dataset_split})")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Limit dataset to max_samples if specified
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset from {original_size} to {len(dataset)} samples")

    print(f"Using {len(dataset)} samples for training")

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

    pass
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    # =============================================================================
    # Set up trainer and training
    # =============================================================================

    # Prepare for training
    FastVisionModel.for_training(model)  # Enable for training!

    # Set up trainer
    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=converted_dataset,  # Use dataset directly!
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(
            model=model, processor=processor
        ),  # Use our custom collator
        args=SFTConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            # Additional optimization settings
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3,
            optim="adamw_torch_fused",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="wandb",
            # Vision-specific settings
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
        ),
    )

    # Rest of the training code remains the same...
    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train the model
    trainer_stats = trainer.train()

    # uncomment to resume from last checkpoint
    # trainer_stats = trainer.train(resume_from_checkpoint=True)

    # Show final memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save the model
    # Create final_weights directory within output_dir
    final_weights_dir = os.path.join(output_dir, "final_weights")
    final_lora_dir = os.path.join(output_dir, "final_lora")
    os.makedirs(final_weights_dir, exist_ok=True)
    os.makedirs(final_lora_dir, exist_ok=True)

    print(f"Saving final lora weights to {final_lora_dir}")
    model.save_pretrained(final_lora_dir)
    processor.save_pretrained(final_weights_dir)

    # Only push to hub if hub_id is provided
    if hub_id:
        print(f"Pushing LoRA weights to Hugging Face Hub as: {hub_id}_lora")
        model.push_to_hub(f"{hub_id}_lora", token=os.environ["HUGGINGFACE_TOKEN"])
        processor.push_to_hub(f"{hub_id}_lora", token=os.environ["HUGGINGFACE_TOKEN"])
    else:
        print("Skipping LoRA weights push to hub (hub_id not provided)")

    print(f"Saving merged model to {final_weights_dir}")
    model.save_pretrained_merged(
        final_weights_dir, processor, save_method="merged_16bit"
    )

    # Only push merged model if hub_id is provided
    if hub_id:
        print(f"Pushing merged model to Hugging Face Hub as: {hub_id}")
        model.push_to_hub_merged(
            hub_id,
            processor,
            token=os.environ["HUGGINGFACE_TOKEN"],
            save_method="merged_16bit",
        )
    else:
        print("Skipping merged model push to hub (hub_id not provided)")

    # Commit the output to the volume
    exp_volume.commit()

    print("Unsloth fine-tuning completed successfully.")

    return {
        "status": "completed",
        "output_dir": output_dir,
        "method": "unsloth",
        "training_time": trainer_stats.metrics["train_runtime"],
        "memory_used": used_memory,
    }


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
    """
    Export and merge LoRA weights with base model.

    This function loads a LoRA fine-tuned model, merges the LoRA weights with the base model,
    and optionally pushes to Hugging Face Hub or saves locally.

    Args:
        lora_model_path: Path to the LoRA weights (can be local path or HF hub ID)
        output_path: Local path to save the merged model (if not pushing to hub)
        hub_model_id: Hugging Face Hub ID to push the merged model to
        push_to_hub: Whether to push the merged model to Hugging Face Hub

    Returns:
        dict: Contains export status and paths
    """
    from unsloth import FastVisionModel
    import os
    import torch

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Disable dynamo for stable operation
    torch._dynamo.config.disable = True

    print(f"Exporting model from {lora_model_path}")

    # Load the LoRA fine-tuned model
    model, processor = FastVisionModel.from_pretrained(
        lora_model_path,
        load_in_4bit=False,
    )

    # Prepare for inference (merges LoRA weights)
    FastVisionModel.for_inference(model)

    if push_to_hub and hub_model_id:
        print(f"Pushing to hub: {hub_model_id}")

        # Save to float16 and push to hub
        model.push_to_hub_merged(
            hub_model_id,
            processor,
            token=os.environ["HUGGINGFACE_TOKEN"],
            save_method="merged_16bit",
        )

        print(f"✓ Pushed to https://huggingface.co/{hub_model_id}")

        # Commit changes to volume
        exp_volume.commit()

        return {
            "status": "completed",
            "lora_model_path": lora_model_path,
            "hub_model_id": hub_model_id,
            "pushed_to_hub": True,
        }
    else:
        # Save locally as merged model
        if output_path is None:
            output_path = f"{lora_model_path}_merged"

        print(f"Saving to: {output_path}")
        os.makedirs(output_path, exist_ok=True)

        model.save_pretrained_merged(output_path, processor, save_method="merged_16bit")
        print(f"✓ Saved to {output_path}")

        # Commit changes to volume
        exp_volume.commit()

        return {
            "status": "completed",
            "lora_model_path": lora_model_path,
            "export_path": output_path,
            "pushed_to_hub": False,
        }


# =============================================================================
# VLLM SERVING CONFIGURATION
# =============================================================================

# Default serving configuration
DEFAULT_SERVE_MODEL = "/data/Finetuned_Gemma_3_4b_it/final_weights"  # Use the base model by default (change to your hub_id after fine-tuning)
SERVE_GPU = "L40S"  # "a100-80gb", "a100-40gb", "l40s"
SERVE_NUM_GPUS = 1
SERVE_GPU_CONFIG = f"{SERVE_GPU}:{SERVE_NUM_GPUS}"
VLLM_PORT = 8000

# CUDA configuration for vLLM
VLLM_CUDA_VERSION = "12.8.1"
VLLM_CUDA_FLAVOR = "devel"
VLLM_CUDA_OS = "ubuntu24.04"
VLLM_CUDA_TAG = f"{VLLM_CUDA_VERSION}-{VLLM_CUDA_FLAVOR}-{VLLM_CUDA_OS}"

# Build vLLM serving image
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
        "regex",
        "sentencepiece",
    )
    .run_commands(
        "uv pip install 'flash-attn>=2.7.1,<=2.8.0' --no-build-isolation --system"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands("python -c 'import torch; print(torch.__version__);'")
)


@app.function(
    image=VLLM_GPU_IMAGE,
    gpu=SERVE_GPU_CONFIG,
    scaledown_window=3 * 60,  # how long should we stay up with no requests? 3 minutes
    secrets=[huggingface_secret],
    volumes=VOLUME_CONFIG,
    max_containers=2,
    timeout=24 * HOURS,
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=8000, startup_timeout=5 * 60)
def serve_vllm():
    """
    Serve a model using vLLM for fast inference.

    Configuration is controlled via module-level constants:
    - DEFAULT_SERVE_MODEL: Model to serve (HF hub ID or local path)
    - VLLM_PORT: Port to serve on
    - SERVE_NUM_GPUS: Number of GPUs to use for tensor parallelism

    Returns:
        Web server endpoint
    """
    import subprocess

    # Set up environment variables
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        DEFAULT_SERVE_MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]

    # Compilation settings - use enforce-eager for faster boot
    cmd += ["--enforce-eager"]

    # GPU configuration
    cmd += ["--tensor-parallel-size", str(SERVE_NUM_GPUS)]
    cmd += ["--gpu-memory-utilization", "0.4"]

    cmd += ["--trust-remote-code"]

    print("Starting vLLM server with command:")
    print(" ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Build evaluation image (CPU-based)
EVAL_IMAGE = (
    ModalImage.debian_slim(python_version="3.12")
    .pip_install(
        "openai",
        "datasets",
        "pillow",
        "numpy",
        "jiwer",
        "nltk",
        "tqdm",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=EVAL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=2 * HOURS,
)
def evaluate_model(
    endpoint_url: str = None,
    model_name: str = "/data/Finetuned_Gemma_3_4b_it/final_weights",
    dataset_name: str = "unsloth/LaTeX_OCR",
    dataset_split: str = "test",
    max_samples: int = 100,
    max_parallel_requests: int = 8,
    temperature: float = 0.1,
    max_tokens: int = 512,
):
    """
    Evaluate a vision-language model on the LaTeX OCR dataset.

    Args:
        endpoint_url: URL of the inference endpoint (e.g., "https://your-endpoint.modal.run/v1").
                     If None, automatically retrieves from deployed serve_vllm function.
        model_name: Model name/path to use for inference
        dataset_name: Name of the dataset to evaluate on
        dataset_split: Dataset split to use
        max_samples: Maximum number of samples to evaluate
        max_parallel_requests: Number of parallel requests to make
        temperature: Temperature for inference
        max_tokens: Maximum tokens to generate

    Returns:
        dict: Contains evaluation metrics and results
    """
    import base64
    import io
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from datasets import load_dataset
    from jiwer import wer, cer
    from tqdm import tqdm
    import time

    # Auto-retrieve endpoint URL if not provided
    if endpoint_url is None:
        try:
            endpoint_url = serve_vllm.get_web_url()
            if endpoint_url:
                endpoint_url = endpoint_url.rstrip("/") + "/v1"
                print(f"Auto-detected endpoint: {endpoint_url}")
            else:
                raise ValueError("serve_vllm endpoint URL not available")
        except Exception as e:
            raise ValueError(
                f"Could not auto-detect endpoint URL: {e}. "
                "Please provide endpoint_url explicitly or ensure serve_vllm is deployed."
            )

    # Load dataset
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Limit to max_samples
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Evaluating {len(dataset)} samples from {dataset_name} on {endpoint_url}")

    # Initialize OpenAI client
    client = OpenAI(base_url=endpoint_url, api_key="EMPTY")

    # Instruction for the model
    instruction = "Write the LaTeX representation for this image."

    def encode_image_to_base64(image):
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def run_inference(sample, idx):
        """Run inference on a single sample."""
        try:
            # Encode image
            image_base64 = encode_image_to_base64(sample["image"])

            # Make request
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": instruction,
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )

            prediction = response.choices[0].message.content.strip()
            ground_truth = sample["text"].strip()

            return {
                "idx": idx,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "success": True,
                "error": None,
            }

        except Exception as e:
            return {
                "idx": idx,
                "prediction": None,
                "ground_truth": sample["text"].strip(),
                "success": False,
                "error": str(e),
            }

    # Run parallel inference
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(run_inference, dataset[i], i): i
            for i in range(len(dataset))
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(dataset), desc="Evaluating", unit="sample") as pbar:
            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                pbar.update(1)

    end_time = time.time()
    total_time = end_time - start_time

    # Sort results by index
    results.sort(key=lambda x: x["idx"])

    successful_results = [r for r in results if r["success"]]
    failed_count = len(results) - len(successful_results)

    if len(successful_results) == 0:
        return {
            "status": "failed",
            "error": "All predictions failed",
            "failed_count": failed_count,
            "total_samples": len(results),
        }

    predictions = [r["prediction"] for r in successful_results]
    ground_truths = [r["ground_truth"] for r in successful_results]

    # Calculate WER (Word Error Rate) and CER (Character Error Rate)
    try:
        word_error_rate = wer(ground_truths, predictions)
        char_error_rate = cer(ground_truths, predictions)
    except Exception:
        word_error_rate = None
        char_error_rate = None

    # Calculate exact match accuracy
    exact_matches = sum(
        1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip()
    )
    exact_match_accuracy = exact_matches / len(successful_results)

    # Calculate average lengths
    avg_pred_length = sum(len(p) for p in predictions) / len(predictions)
    avg_gt_length = sum(len(g) for g in ground_truths) / len(ground_truths)

    # Print concise results
    print(f"\n{'=' * 80}")
    print(
        f"Results: {len(successful_results)}/{len(results)} successful ({len(successful_results) / len(results) * 100:.1f}%)"
    )
    print(
        f"Exact Match: {exact_match_accuracy * 100:.1f}% | CER: {char_error_rate * 100:.1f}% | WER: {word_error_rate * 100:.1f}%"
        if char_error_rate and word_error_rate
        else f"Exact Match: {exact_match_accuracy * 100:.1f}%"
    )
    print(f"Time: {total_time:.1f}s ({len(results) / total_time:.1f} samples/s)")
    print(f"{'=' * 80}")

    return {
        "status": "completed",
        "endpoint_url": endpoint_url,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "total_samples": len(results),
        "successful_samples": len(successful_results),
        "failed_samples": failed_count,
        "success_rate": len(successful_results) / len(results),
        "metrics": {
            "exact_match_accuracy": exact_match_accuracy,
            "character_error_rate": char_error_rate,
            "word_error_rate": word_error_rate,
        },
        "statistics": {
            "avg_prediction_length": avg_pred_length,
            "avg_ground_truth_length": avg_gt_length,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(results),
            "throughput_samples_per_second": len(results) / total_time,
        },
        "examples": [
            {
                "ground_truth": r["ground_truth"],
                "prediction": r["prediction"],
                "match": r["prediction"].strip() == r["ground_truth"].strip(),
            }
            for r in successful_results[:10]
        ],
    }
