from dataclasses import dataclass, field
from typing import Optional

import torch

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    learning_rate: Optional[float] = field(default=0.0002)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=4096)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="CognitiveLab/Hindi-Instruct-Gemma-Prompt-formate",
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    # max_steps: int = field(default=1000, metadata={"help": "How many optimizer update steps to take"})
    num_train_epochs: int = field(default=1, metadata={"help": "How many epochs you want to train it for"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=30, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="Gemma-Hindi-Instruct",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# def formatting_func(example):
#     text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
#     return text

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = "google/gemma-7b-it"

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4"
# )

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    # quantization_config=quantization_config, 
    torch_dtype=torch.float32,
    device_map={"": 0},
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


lora_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout
)

train_dataset = load_dataset(script_args.dataset_name, split="train[:5%]")

# TODO: make that configurable
YOUR_HF_USERNAME = "CognitiveLab"
output_dir = f"{YOUR_HF_USERNAME}/gemma-hindi-instruct"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_grad_norm=script_args.max_grad_norm,
    # max_steps=script_args.max_steps,
    num_train_epochs=script_args.num_train_epochs,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    report_to=script_args.report_to,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=lora_config,
    packing=script_args.packing,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=script_args.max_seq_length,
)

trainer.train()

logger.info("Training stage completed")
peft_model = script_args.output_dir
trainer.model.save_pretrained(peft_model)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, peft_model)
merged_model= merged_model.merge_and_unload()
logger.info("Training stage completed")


merged_model_name = str(peft_model)+"_merged"
# Save the merged model
logger.info("Merging the model with the PEFT adapter")
merged_model.save_pretrained(merged_model_name,safe_serialization=True)
tokenizer.save_pretrained("merged_model")


logger.info("Pushing the model to Huggingface hub")
try:
    merged_model.push_to_hub(script_args.output_dir, use_temp_dir=False)
    tokenizer.push_to_hub(script_args.output_dir, use_temp_dir=False)
except Exception as e:
    logger.info(f"Error while pushing to huggingface Hub: {e}")

logger.info("Training stage completed")