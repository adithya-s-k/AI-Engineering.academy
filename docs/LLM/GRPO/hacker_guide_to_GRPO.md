# Hacker Guide to GRPO

## Introduction

Group Relative Policy Optimization (GRPO) represents a significant advancement in the field of Large Language Model (LLM) fine-tuning. Traditional reinforcement learning approaches like Proximal Policy Optimization (PPO) have been widely used for fine-tuning LLMs, but they often come with substantial computational overhead due to their requirement for value function estimation and complex reward calculation mechanisms.
GRPO, introduced by DeepSeek, offers an elegant solution to these challenges. Instead of relying on a value function, GRPO uses a group-based approach to estimate advantages, which significantly reduces both computational complexity and memory requirements. This makes it particularly attractive for researchers and practitioners working with limited computational resources.
The key innovation of GRPO lies in its group-relative reward mechanism. Rather than comparing policy updates against an absolute baseline, it evaluates performance relative to a group of samples. This approach not only simplifies the training process but also provides more stable and efficient learning, especially for tasks requiring complex reasoning like mathematical problem-solving.

Key Benefits of GRPO:

- Reduced computational overhead compared to traditional PPO
- No need for value function estimation
- More efficient memory utilization
- Improved stability in training
- Particularly effective for mathematical reasoning tasks
- Simpler implementation and maintenance

This guide will walk you through implementing GRPO for fine-tuning language models, with a focus on practical applications and efficient deployment.

## Initial Setup

First, install the required dependencies:

```bash
pip install rich trl peft datasets transformers evaluate
pip install accelerate
pip install pydantic
pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
pip install wandb
pip install --upgrade "jinja2>=3.1.0"
```

Reference implementation adapted from: [Github gist](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)

## Try on [Notebook](/docs/LLM/GRPO/Qwen_0_5b__GRPO.ipynb)

<a href="https://colab.research.google.com/github/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/GRPO/Qwen_0_5b__GRPO.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Implementation Details

Create a file named `grpo.py` with the following implementation:

```python
# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# Uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

#model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir="outputs/Qwen-1.5B-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    beta = 0.01,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=786,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="wandb",
    log_on_each_node=False,
    # use_vllm=True,
    # vllm_device="cuda:0",
    # vllm_gpu_memory_utilization= 0.4
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Use PEFT at your own risk; not working for multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    #peft_config=peft_config
)
trainer.train()
```

## Multi-GPU Setup Guide

1. First, run `nvidia-smi` to check the number of available GPUs on your system.

2. Ensure `accelerate` is installed:

```bash
pip install accelerate
```

3. Configuration Notes:

   - The provided configuration works well with Qwen 0.5B, Qwen 1B, and Llama 1B models on 2 A100 gpus
   - Using VLLM provides significant speed improvements while maintaining similar training quality
   - To enable VLLM, uncomment these lines in the GRPOConfig:

   ```python
   use_vllm=True,
   vllm_device="cuda:0",
   vllm_gpu_memory_utilization=0.4
   ```

4. VLLM Memory Usage Guidelines:

   - For models under 1 billion parameters on an A100 GPU, you can lower `vllm_gpu_memory_utilization` to 0.2
   - For larger models (>3B parameters), test and adjust the utilization accordingly

5. For multi-GPU training, use `accelerate`:

```bash
accelerate launch grpo.py
```

## Configuration and Parameters Guide

### Initial Setup and Authentication

Before running the training, you'll need to set up authentication for both Weights & Biases (wandb) and Hugging Face:

```bash
# Login to Weights & Biases
wandb login

# Login to Hugging Face
huggingface-cli login
```

### Understanding GRPO Parameters

Let's examine each parameter in detail:

1. **Basic Configuration**

   - `output_dir`: Directory where model checkpoints and logs will be saved
   - `run_name`: Name of your training run, used for logging and organization
   - `beta`: Controls the strength of the policy update. Values between 0.01 and 0.05 typically yield better results. This parameter is crucial for training stability.

2. **Optimization Parameters**

   - `learning_rate`: Set to 5e-6 by default. This relatively small learning rate helps prevent destructive updates
   - `adam_beta1` (0.9) and `adam_beta2` (0.99): Momentum parameters for the Adam optimizer
   - `weight_decay` (0.1): Helps prevent overfitting
   - `warmup_ratio` (0.1): Portion of training used for learning rate warmup
   - `lr_scheduler_type`: Uses 'cosine' schedule for smooth learning rate decay

3. **Training Configuration**
   - `logging_steps`: Set to 1 for maximum visibility into training progress
   - `bf16`: Enable bfloat16 training for better memory efficiency
   - `per_device_train_batch_size`: Start with 1 and adjust based on memory
   - `gradient_accumulation_steps`: Set to 4 by default, can be adjusted for memory constraints
   - `num_generations`: Number of generations per prompt (16 by default)
   - `max_prompt_length`: Maximum token length for input prompts (256 by default)
   - `max_completion_length`: Maximum token length for completions (786 by default)
   - `num_train_epochs`: Number of training epochs
   - `save_steps`: Frequency of saving checkpoints
   - `max_grad_norm`: Gradient clipping threshold

### Memory Management and Optimization

When encountering CUDA out of memory errors, consider these adjustments in order of priority:

1. Reduce batch size (`per_device_train_batch_size`)
2. Decrease gradient accumulation steps
3. Lower the number of generations (`num_generations`)

However, note that higher numbers of generations often lead to better and faster results. Finding the right balance for your hardware is key.

### Input/Output Length Considerations

Adjust `max_prompt_length` and `max_completion_length` based on your specific use case:

```python
# For short Q&A tasks
max_prompt_length = 128
max_completion_length = 256

# For complex reasoning tasks
max_prompt_length = 512
max_completion_length = 1024

# For long-form content generation
max_prompt_length = 1024
max_completion_length = 2048
```

### Reward Function Design

Reward modeling is a critical aspect of GRPO training. Poor reward functions can lead to reward hacking, where the model optimizes for the reward in unintended ways. Here are key principles for reward function design:

1. **Magnitude Balancing**: Ensure reward magnitudes are appropriately scaled

   ```python
   def balanced_reward_func(completions, **kwargs) -> list[float]:
       # Primary task reward: scale of 1.0
       primary_reward = calculate_primary_reward()

       # Secondary objectives: smaller scale
       format_reward = check_format() * 0.2
       quality_reward = assess_quality() * 0.3

       return primary_reward + format_reward + quality_reward
   ```

2. **Reward Composition**: Combine multiple objectives carefully

   ```python
   def composite_reward_func(completions, **kwargs) -> list[float]:
       rewards = []
       for completion in completions:
           # Base reward for task completion
           reward = assess_task_completion(completion)

           # Penalties for unwanted behaviors
           if contains_repetition(completion):
               reward *= 0.8

           # Bonus for exceptional quality
           if meets_quality_threshold(completion):
               reward *= 1.2

           rewards.append(reward)
       return rewards
   ```

The optimal reward function configuration often requires intuition and empirical testing. Start with simple reward functions and gradually add complexity while monitoring training dynamics.
