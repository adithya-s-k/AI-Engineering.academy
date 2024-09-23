Title: A Beginner’s Guide to Fine-Tuning Mistral 7B Instruct Model

URL Source: https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe

Published Time: 2023-10-06T18:30:13.121Z

Markdown Content:
Fine-Tuning for Code Generation Using a Single Google Colab Notebook
--------------------------------------------------------------------

[![Image 1: Adithya S K](https://miro.medium.com/v2/resize:fill:88:88/1*w1_VSVDg5oqt19oTB4MAMg.jpeg)](https://adithyask.medium.com/?source=post_page-----0f39647b20fe--------------------------------)

> _Updated : 10th December 2023_

Fine-tuning a state-of-the-art language model like Mistral 7B Instruct can be an exciting journey. This guide will walk you through the process step by step, from setting up your environment to fine-tuning the model for your specific task. Whether you’re a seasoned machine learning practitioner or a newcomer to the field, this beginner-friendly tutorial will help you harness the power of Mistral 7B for your projects.

Meet Mistral 7B Instruct
------------------------

The team at [MistralAI](https://mistral.ai/news/announcing-mistral-) has created an exceptional language model called Mistral 7B Instruct. It has consistently delivered outstanding results in a range of benchmarks, which positions it as an ideal option for natural language generation and understanding. This guide will concentrate on how to fine-tune the model for coding purposes, but the methodology can effectively be applied to other tasks.

Colab Notebook to Finetuning Mistral-7b-Instruct
------------------------------------------------

Code has been updated on December 10th , 2023

Prerequisites
-------------

Before diving into the fine-tuning process, make sure you have the following prerequisites in place:

1.  **GPU**: While this tutorial can run on a free Google Colab notebook with a GPU, it’s recommended to use more powerful GPUs like V100 or A100 for better performance.
2.  **Python Packages**: Ensure you have the required Python packages installed. You can run the following commands to install them:

!pip install -q torch  
!pip install -q git+https://github.com/huggingface/transformers #huggingface transformers for downloading models weights  
!pip install -q datasets #huggingface datasets to download and manipulate datasets  
!pip install -q peft #Parameter efficient finetuning - for qLora Finetuning  
!pip install -q bitsandbytes #For Model weights quantisation  
!pip install -q trl #Transformer Reinforcement Learning - For Finetuning using Supervised Fine-tuning  
!pip install -q wandb -U #Used to monitor the model score during training

1.  **Hugging Face Hub Account**: You’ll need an account on the Hugging Face Model Hub. You can sign up [here](https://huggingface.co/join).

Getting Started
---------------

Let’s start by checking if your GPU is correctly detected:

!nvidia-smi

If your GPU is not recognized or you encounter CUDA out-of-memory errors during fine-tuning, consider using a more powerful GPU.

Loading Required Libraries
--------------------------

We’ll load the necessary Python libraries for our fine-tuning process:

import json  
import pandas as pd  
import torch  
from datasets import Dataset, load\_dataset  
from huggingface\_hub import notebook\_login  
from peft import LoraConfig, PeftModel  
from transformers import (  
    AutoModelForCausalLM,  
    AutoTokenizer,  
    BitsAndBytesConfig,  
    TrainingArguments,  
    pipeline,  
    logging,  
)  
from trl import SFTTrainer

Logging into Hugging Face Hub
-----------------------------

Log in to the Hugging Face Model Hub using your credentials:

notebook\_login()

Loading the Dataset
-------------------

For this tutorial, we will fine-tune Mistral 7B Instruct for code generation.

we will be using this [dataset](https://huggingface.co/datasets/TokenBender/code_instructions_122k_alpaca_style) which is curated by [TokenBender (e/xperiments)](https://twitter.com/4evaBehindSOTA) which is a awesome data for finetuning model for code generation. It follows the alpaca style of instructions which is an excellent starting point for this task. The dataset structure should resemble the following:

{  
    "instruction": "Create a function to calculate the sum of a sequence of integers.",  
    "input":"\[1, 2, 3, 4, 5\]",  
    "output": "# Python code def sum\_sequence(sequence): sum = 0 for num in sequence: sum += num return sum"  
}

now lets load the dataset using huggingfaces datasets library

\# Load your dataset (replace 'your\_dataset\_name' and 'split\_name' with your actual dataset information)  
\# dataset = load\_dataset("your\_dataset\_name", split="split\_name")  
dataset = load\_dataset("TokenBender/code\_instructions\_122k\_alpaca\_style", split="train")

Formatting the Dataset
----------------------

Now, let’s format the dataset in the required [Mistral-7B-Instruct-v0.1 format](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).

> _Many tutorial and blogs skip over this part but i feel this is a really important step._

We’ll put each instruction and input pair between `[INST]` and `[/INST]` output after that, like this:

<s\>\[INST\] What is your favourite condiment? \[/INST\]  
Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s\>

You can use the following code to process your dataset and create a JSONL file in the correct format:

\# this function is used to output the right formate for each row in the dataset  
def create\_text\_row(instruction, output, input):  
    text\_row = f"""<s\>\[INST\] {instruction} here are the inputs {input} \[/INST\] \\\\n {output} </s\>"""  
    return text\_row\# interate over all the rows formate the dataset and store it in a jsonl file  
def process\_jsonl\_file(output\_file\_path):  
    with open(output\_file\_path, "w") as output\_jsonl\_file:  
        for item in dataset:  
            json\_object = {  
                "text": create\_text\_row(item\["instruction"\], item\["input"\] ,item\["output"\]),  
                "instruction": item\["instruction"\],  
                "input": item\["input"\],  
                "output": item\["output"\]  
            }  
            output\_jsonl\_file.write(json.dumps(json\_object) + "\\\\n")

\# Provide the path where you want to save the formatted dataset  
process\_jsonl\_file("./training\_dataset.jsonl")

After Formatting
----------------

{  
"text":"<s\>\[INST\] Create a function to calculate the sum of a sequence of integers. here are the inputs \[1, 2, 3, 4, 5\] \[/INST\]  
\# Python code def sum\_sequence(sequence): sum = 0 for num in sequence: sum += num return sum</s\>",  
"instruction":"Create a function to calculate the sum of a sequence of integers",  
"input":"\[1, 2, 3, 4, 5\]",  
"output":"# Python code def sum\_sequence(sequence): sum = 0 for num in sequence: sum += num return sum"  
}

While using SFT([Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)) to finetune we will be only passing in the “text” column of the dataset for finetuning.

Loading the Training Dataset
----------------------------

Now, let’s load the training dataset from the JSONL file we created:

train\_dataset = load\_dataset('json', data\_files='./training\_dataset.jsonl' , split='train')

Setting Model Parameters
------------------------

We need to set various parameters for our fine-tuning process, including QLoRA (Quantization LoRA) parameters, bitsandbytes parameters, and training arguments:

new\_model = "mistralai-Code-Instruct" #set the name of the new model################################################################################  
\# QLoRA parameters  
################################################################################

\# LoRA attention dimension  
lora\_r = 64

\# Alpha parameter for LoRA scaling  
lora\_alpha = 16

\# Dropout probability for LoRA layers  
lora\_dropout = 0.1

################################################################################  
\# bitsandbytes parameters  
################################################################################

\# Activate 4-bit precision base model loading  
use\_4bit = True

\# Compute dtype for 4-bit base models  
bnb\_4bit\_compute\_dtype = "float16"

\# Quantization type (fp4 or nf4)  
bnb\_4bit\_quant\_type = "nf4"

\# Activate nested quantization for 4-bit base models (double quantization)  
use\_nested\_quant = False

################################################################################  
\# TrainingArguments parameters  
################################################################################

\# Output directory where the model predictions and checkpoints will be stored  
output\_dir = "./results"

\# Number of training epochs  
num\_train\_epochs = 1

\# Enable fp16/bf16 training (set bf16 to True with an A100)  
fp16 = False  
bf16 = False

\# Batch size per GPU for training  
per\_device\_train\_batch\_size = 4

\# Batch size per GPU for evaluation  
per\_device\_eval\_batch\_size = 4

\# Number of update steps to accumulate the gradients for  
gradient\_accumulation\_steps = 1

\# Enable gradient checkpointing  
gradient\_checkpointing = True

\# Maximum gradient normal (gradient clipping)  
max\_grad\_norm = 0.3

\# Initial learning rate (AdamW optimizer)  
learning\_rate = 2e-4

\# Weight decay to apply to all layers except bias/LayerNorm weights  
weight\_decay = 0.001

\# Optimizer to use  
optim = "paged\_adamw\_32bit"

\# Learning rate schedule (constant a bit better than cosine)  
lr\_scheduler\_type = "constant"

\# Number of training steps (overrides num\_train\_epochs)  
max\_steps = -1

\# Ratio of steps for a linear warmup (from 0 to learning rate)  
warmup\_ratio = 0.03

\# Group sequences into batches with same length  
\# Saves memory and speeds up training considerably  
group\_by\_length = True

\# Save checkpoint every X updates steps  
save\_steps = 25

\# Log every X updates steps  
logging\_steps = 25

################################################################################  
\# SFT parameters  
################################################################################

\# Maximum sequence length to use  
max\_seq\_length = None

\# Pack multiple short examples in the same input sequence to increase efficiency  
packing = False

\# Load the entire model on the GPU 0  
device\_map = {"": 0}

Loading the Base Model
----------------------

Let’s load the Mistral 7B Instruct base model:

model\_name = "mistralai/Mistral-7B-Instruct-v0.1"\# Load the base model with QLoRA configuration  
compute\_dtype = getattr(torch, bnb\_4bit\_compute\_dtype)

bnb\_config = BitsAndBytesConfig(  
    load\_in\_4bit=use\_4bit,  
    bnb\_4bit\_quant\_type=bnb\_4bit\_quant\_type,  
    bnb\_4bit\_compute\_dtype=compute\_dtype,  
    bnb\_4bit\_use\_double\_quant=use\_nested\_quant,  
)

base\_model = AutoModelForCausalLM.from\_pretrained(  
    model\_name,  
    quantization\_config=bnb\_config,  
    device\_map={"": 0}  
)

base\_model.config.use\_cache = False  
base\_model.config.pretraining\_tp = 1

\# Load MitsralAi tokenizer  
tokenizer = AutoTokenizer.from\_pretrained(model\_name, trust\_remote\_code=True)  
tokenizer.pad\_token = tokenizer.eos\_token  
tokenizer.padding\_side = "right"pyt

Base model Inference
--------------------

eval\_prompt = """Print hello world in python c and c++"""\# import random  
model\_input = tokenizer(eval\_prompt, return\_tensors="pt").to("cuda")

model.eval()  
with torch.no\_grad():  
    print(tokenizer.decode(model.generate(\*\*model\_input, max\_new\_tokens=256, pad\_token\_id=2)\[0\], skip\_special\_tokens=True))Fine-Tuning with qLora

The results from the base model tend to be of poor quality and doesn’t always generate sytactically correct code

Fine-Tuning with qLora and Supervised Finetuning
------------------------------------------------

We’re ready to fine-tune our model using qLora. For this tutorial, we’ll use the `SFTTrainer` from the `trl` library for supervised fine-tuning. Ensure that you've installed the `trl` library as mentioned in the prerequisites.

\# Set LoRA configuration  
peft\_config = LoraConfig(  
    lora\_alpha=lora\_alpha,  
    lora\_dropout=lora\_dropout,  
    r=lora\_r,  
    target\_modules=\[  
        "q\_proj",  
        "k\_proj",  
        "v\_proj",  
        "o\_proj",  
        "gate\_proj",  
        "up\_proj",  
        "down\_proj",  
        "lm\_head",  
    \],  
    bias="none",  
    task\_type="CAUSAL\_LM",  
)\# Set training parameters  
training\_arguments = TrainingArguments(  
    output\_dir=output\_dir,  
    num\_train\_epochs=num\_train\_epochs,  
    per\_device\_train\_batch\_size=per\_device\_train\_batch\_size,  
    gradient\_accumulation\_steps=gradient\_accumulation\_steps,  
    optim=optim,  
    save\_steps=save\_steps,  
    logging\_steps=logging\_steps,  
    learning\_rate=learning\_rate,  
    weight\_decay=weight\_decay,  
    fp16=fp16,  
    bf16=bf16,  
    max\_grad\_norm=max\_grad\_norm,  
    max\_steps=100, # the total number of training steps to perform  
    warmup\_ratio=warmup\_ratio,  
    group\_by\_length=group\_by\_length,  
    lr\_scheduler\_type=lr\_scheduler\_type,  
    report\_to="tensorboard"  
)

\# Initialize the SFTTrainer for fine-tuning  
trainer = SFTTrainer(  
    model=base\_model,  
    train\_dataset=train\_dataset,  
    peft\_config=peft\_config,  
    dataset\_text\_field="text",  
    max\_seq\_length=max\_seq\_length,  # You can specify the maximum sequence length here  
    tokenizer=tokenizer,  
    args=training\_arguments,  
    packing=packing,  
)

Lets start the training process
-------------------------------

\# Start the training process  
trainer.train()\# Save the fine-tuned model  
trainer.model.save\_pretrained(new\_model)

Inference with Fine-Tuned Model
-------------------------------

Now that we have fine-tuned our model, let’s test its performance with some code generation tasks. Replace `eval_prompt` with your code generation prompt:

eval\_prompt = """Print hello world in python c and c++"""model\_input = tokenizer(eval\_prompt, return\_tensors="pt").to("cuda")  
model.eval()  
with torch.no\_grad():  
    generated\_code = tokenizer.decode(model.generate(\*\*model\_input, max\_new\_tokens=256, pad\_token\_id=2)\[0\], skip\_special\_tokens=True)  
print(generated\_code)

Merge and Share
---------------

After fine-tuning, if you want to merge the model with LoRA weights or share it with the Hugging Face Model Hub, you can do so. This step is optional and depends on your specific use case.

\# Merge the model with LoRA weights  
base\_model = AutoModelForCausalLM.from\_pretrained(  
    model\_name,  
    low\_cpu\_mem\_usage=True,  
    return\_dict=True,  
    torch\_dtype=torch.float16,  
    device\_map={"": 0},  
)  
merged\_model= PeftModel.from\_pretrained(base\_model, new\_model)  
merged\_model= model.merge\_and\_unload()\# Save the merged model  
merged\_model.save\_pretrained("merged\_model",safe\_serialization=True)  
tokenizer.save\_pretrained("merged\_model")

\# Merge the model with LoRA weights  
base\_model = AutoModelForCausalLM.from\_pretrained(  
    model\_name,  
    low\_cpu\_mem\_usage=True,  
    return\_dict=True,  
    torch\_dtype=torch.float16,  
    device\_map={"": 0},  
)  
merged\_model= PeftModel.from\_pretrained(base\_model, new\_model)  
merged\_model= merged\_model.merge\_and\_unload()

\# Save the merged model  
merged\_model.save\_pretrained("merged\_model",safe\_serialization=True)  
tokenizer.save\_pretrained("merged\_model")

Test the merged model
---------------------

from random import randrange  
sample = train\_dataset \[randrange(len(train\_dataset ))\]prompt = f"""<s\>   
{sample\['instruction'\]}  
{sample\['input'\]}  
\[INST\]

"""

input\_ids = tokenizer(prompt, return\_tensors="pt", truncation=True).input\_ids.cuda()  
\# with torch.inference\_mode():  
outputs = merged\_model.generate(input\_ids=input\_ids, max\_new\_tokens=100, do\_sample=True, top\_p=0.9,temperature=0.5)

print(f"Prompt:\\n{prompt}\\n")  
print(f"\\nGenerated instruction:\\n{tokenizer.batch\_decode(outputs.detach().cpu().numpy(), skip\_special\_tokens=True)\[0\]\[len(prompt):\]}")  
print(f"\\nGround truth:\\n{sample\['output'\]}")

And that’s it! You’ve successfully fine-tuned Mistral 7B Instruct for code generation. You can adapt this process for various natural language understanding and generation tasks. Keep exploring and experimenting with Mistral 7B to unlock its full potential for your projects.

All the code will be available on my github. Do drop by and give a follow and a star

I also post content about Generative AI | LLMs | Stable Diffusion and what i have been working on twitter — [AdithyaSK (@adithya\_s\_k) / X](https://twitter.com/adithya_s_k)