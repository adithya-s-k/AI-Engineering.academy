Title: A Beginner’s Guide to Fine-Tuning Gemma - Adithya S K - Medium

URL Source: https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-gemma-0444d46d821c

Published Time: 2024-02-21T20:13:11.688Z

Markdown Content:
[![Image 1: Adithya S K](https://miro.medium.com/v2/resize:fill:88:88/1*w1_VSVDg5oqt19oTB4MAMg.jpeg)](https://adithyask.medium.com/?source=post_page-----0444d46d821c--------------------------------)

A Comprehensive Guide to Fine-Tuning Gemma

Fine-tuning a state-of-the-art language model like Gemma can be an exciting journey. This guide will walk you through the process step by step, from setting up your environment to fine-tuning the model for your specific task. Whether you’re a seasoned machine learning practitioner or a newcomer to the field, this beginner-friendly tutorial will help you harness the power of Gemma for your projects.

Meet Gemma
----------

a family of lightweight, state-of-the art open models built from the research and technology used to create Gemini models. Gemma models demonstrate strong performance across academic benchmarks for language understanding, reasoning, and safety. We release two sizes of models (2 billion and 7 billion parameters), and provide both pretrained and fine-tuned checkpoints. Gemma outperforms similarly sized open models on 11 out of 18 text-based tasks, and we present comprehensive evaluations of safety and responsibility aspects of the models, alongside a detailed description of model development. We believe the responsible release of LLMs is critical for improving the safety of frontier models, and for enabling the next wave of LLM innovations.

Colab Notebook to Finetuning
----------------------------

Github Repository
-----------------

Prerequisites
-------------

Before delving into the fine-tuning process, ensure that you have the following prerequisites in place:

1\. **GPU**: [gemma-2b](https://huggingface.co/google/gemma-2b) — can be finetuned on T4(free google colab) while [gemma-7b](https://huggingface.co/google/gemma-7b) requires an A100 GPU.

2\. **Python Packages**: Ensure that you have the necessary Python packages installed. You can use the following commands to install them:

Let’s begin by checking if your GPU is correctly detected:

!pip3 install -q -U bitsandbytes==0.42.0  
!pip3 install -q -U peft==0.8.2  
!pip3 install -q -U trl==0.7.10  
!pip3 install -q -U accelerate==0.27.1  
!pip3 install -q -U datasets==2.17.0  
!pip3 install -q -U transformers==4.38.0

Hugging Face Hub Account: You’ll need an account on the Hugging Face Model Hub. You can sign up [here](https://huggingface.co/join).

Getting Started
---------------

Checking GPU
------------

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

Loading the Model
-----------------

model\_id = "google/gemma-7b-it"  
\# model\_id = "google/gemma-7b"  
\# model\_id = "google/gemma-2b-it"  
\# model\_id = "google/gemma-2b"model = AutoModelForCausalLM.from\_pretrained(model\_id, quantization\_config=bnb\_config, device\_map={"":0})  
tokenizer = AutoTokenizer.from\_pretrained(model\_id, add\_eos\_token=True)

Loading the Dataset
-------------------

For this tutorial, we will fine-tune Gemma Instruct for code generation.

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

Now, let’s format the dataset in the required [gemma instruction formate](https://huggingface.co/google/gemma-7b-it).

> _Many tutorials and blogs skip over this part, but I feel this is a really important step._

<start\_of\_turn\>user What is your favorite condiment? <end\_of\_turn\>  
<start\_of\_turn\>model Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavor to whatever I'm cooking up in the kitchen!<end\_of\_turn\>

You can use the following code to process your dataset and create a JSONL file in the correct format:

def generate\_prompt(data\_point):  
    """Gen. input text based on a prompt, task instruction, (context info.), and answer:param data\_point: dict: Data point  
    :return: dict: tokenzed prompt  
    """  
    prefix\_text = 'Below is an instruction that describes a task. Write a response that ' \\\\  
               'appropriately completes the request.\\\\n\\\\n'  
    # Samples with additional context into.  
    if data\_point\['input'\]:  
        text = f"""<start\_of\_turn\>user {prefix\_text} {data\_point\["instruction"\]} here are the inputs {data\_point\["input"\]} <end\_of\_turn\>\\\\n<start\_of\_turn\>model{data\_point\["output"\]} <end\_of\_turn\>"""  
    # Without  
    else:  
        text = f"""<start\_of\_turn\>user {prefix\_text} {data\_point\["instruction"\]} <end\_of\_turn\>\\\\n<start\_of\_turn\>model{data\_point\["output"\]} <end\_of\_turn\>"""  
    return text

\# add the "prompt" column in the dataset  
text\_column = \[generate\_prompt(data\_point) for data\_point in dataset\]  
dataset = dataset.add\_column("prompt", text\_column)

We'll need to tokenize our data so the model can understand.

dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here  
dataset = dataset.map(lambda samples: tokenizer(samples\["prompt"\]), batched=True)

Split dataset into 90% for training and 10% for testing

dataset = dataset.train\_test\_split(test\_size=0.2)  
train\_data = dataset\["train"\]  
test\_data = dataset\["test"\]

After Formatting, We should get something like this
---------------------------------------------------

{  
"text":"<start\_of\_turn\>user Create a function to calculate the sum of a sequence of integers. here are the inputs \[1, 2, 3, 4, 5\] <end\_of\_turn\>  
<start\_of\_turn\>model # Python code def sum\_sequence(sequence): sum = 0 for num in sequence: sum += num return sum <end\_of\_turn\>",  
"instruction":"Create a function to calculate the sum of a sequence of integers",  
"input":"\[1, 2, 3, 4, 5\]",  
"output":"# Python code def sum\_sequence(sequence): sum = 0 for num in,  
 sequence: sum += num return sum",  
"prompt":"<start\_of\_turn\>user Create a function to calculate the sum of a sequence of integers. here are the inputs \[1, 2, 3, 4, 5\] <end\_of\_turn\>  
<start\_of\_turn\>model # Python code def sum\_sequence(sequence): sum = 0 for num in sequence: sum += num return sum <end\_of\_turn\>"  
}  

While using SFT ([**Supervised Fine-tuning Trainer**](https://huggingface.co/docs/trl/main/en/sft_trainer)) for fine-tuning, we will be only passing in the “text” column of the dataset for fine-tuning.

Setting Model Parameters and Lora
---------------------------------

We need to set various parameters for our fine-tuning process, including QLoRA (Quantization LoRA) parameters, bitsandbytes parameters, and training arguments:

Apply Lora
----------

Here comes the magic with peft! Let’s load a PeftModel and specify that we are going to use low-rank adapters (LoRA) using get\_peft\_model utility function and the prepare\_model\_for\_kbit\_training method from PEFT.

Here is a tweet on how to pick the best Lora config

from peft import LoraConfig, get\_peft\_model  
lora\_config = LoraConfig(  
    r=64,  
    lora\_alpha=32,  
    target\_modules=\['o\_proj', 'q\_proj', 'up\_proj', 'v\_proj', 'k\_proj', 'down\_proj', 'gate\_proj'\],  
    lora\_dropout=0.05,  
    bias="none",  
    task\_type="CAUSAL\_LM"  
)   
model = get\_peft\_model(model, lora\_config)

Calculating the number of trainable parameters

trainable, total = model.get\_nb\_trainable\_parameters()  
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total\*100:.4f}%")

> expected output → Trainable: 200015872 | total: 8737696768 | Percentage: 2.2891%

Fine-Tuning with qLora and Supervised Finetuning
------------------------------------------------

We’re ready to fine-tune our model using qLora. For this tutorial, we’ll use the `SFTTrainer` from the `trl` library for supervised fine-tuning. Ensure that you've installed the `trl` library as mentioned in the prerequisites.

#new code using SFTTrainer  
import transformersfrom trl import SFTTrainer

tokenizer.pad\_token = tokenizer.eos\_token  
torch.cuda.empty\_cache()  
trainer = SFTTrainer(  
    model=model,  
    train\_dataset=train\_data,  
    eval\_dataset=test\_data,  
    dataset\_text\_field="prompt",  
    peft\_config=lora\_config,  
    args=transformers.TrainingArguments(  
        per\_device\_train\_batch\_size=1,  
        gradient\_accumulation\_steps=4,  
        warmup\_steps=0.03,  
        max\_steps=100,  
        learning\_rate=2e-4,  
        logging\_steps=1,  
        output\_dir="outputs",  
        optim="paged\_adamw\_8bit",  
        save\_strategy="epoch",  
    ),  
    data\_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),  
)

Lets start the training process
-------------------------------

\# Start the training process  
trainer.train()new\_model = "gemma-Code-Instruct-Finetune-test" #Name of the model you will be pushing to huggingface model hub  
\# Save the fine-tuned model  
trainer.model.save\_pretrained(new\_model)

Merge and Share
---------------

After fine-tuning, if you want to merge the model with LoRA weights or share it with the Hugging Face Model Hub, you can do so. This step is optional and depends on your specific use case.

\# Merge the model with LoRA weights  
base\_model = AutoModelForCausalLM.from\_pretrained(  
    model\_id,  
    low\_cpu\_mem\_usage=True,  
    return\_dict=True,  
    torch\_dtype=torch.float16,  
    device\_map={"": 0},  
)  
merged\_model= PeftModel.from\_pretrained(base\_model, new\_model)  
merged\_model= merged\_model.merge\_and\_unload()\# Save the merged model  
merged\_model.save\_pretrained("merged\_model",safe\_serialization=True)  
tokenizer.save\_pretrained("merged\_model")  
tokenizer.pad\_token = tokenizer.eos\_token  
tokenizer.padding\_side = "right"

\# Push the model and tokenizer to the Hugging Face Model Hub  
merged\_model.push\_to\_hub(new\_model, use\_temp\_dir=False)  
tokenizer.push\_to\_hub(new\_model, use\_temp\_dir=False)

Test the merged model
---------------------

def get\_completion(query: str, model, tokenizer) -\> str:  
  device = "cuda:0"  
  prompt\_template = """  
  <start\_of\_turn\>user  
  Below is an instruction that describes a task. Write a response that appropriately completes the request.  
  {query}  
  <end\_of\_turn\>\\\\n<start\_of\_turn\>model"""  
  prompt = prompt\_template.format(query=query)  
  encodeds = tokenizer(prompt, return\_tensors="pt", add\_special\_tokens=True)  
  model\_inputs = encodeds.to(device)  
  generated\_ids = model.generate(\*\*model\_inputs, max\_new\_tokens=1000, do\_sample=True, pad\_token\_id=tokenizer.eos\_token\_id)  
  # decoded = tokenizer.batch\_decode(generated\_ids)  
  decoded = tokenizer.decode(generated\_ids\[0\], skip\_special\_tokens=True)  
  return (decoded)

result = get\_completion(query="code the fibonacci series in python using reccursion", model=merged\_model, tokenizer=tokenizer)  
print(result)

And that’s it! You’ve successfully fine-tuned Gemma Instruct for code generation. You can adapt this process for various natural language understanding and generation tasks. Keep exploring and experimenting with Gemma to unlock its full potential for your projects.

Happy Fine-Tuning!!

> _If you found this post valuable, make sure to follow me for more insightful content. I frequently write about the practical applications of Generative AI, LLMs, Stable Diffusion, and explore the broader impacts of AI on society._
> 
> _Let’s stay connected on_ [_Twitter_](https://twitter.com/adithya_s_k)_. I’d love to engage in discussions with you._
> 
> _If you’re not a Medium member yet and wish to support writers like me, consider signing up through my referral link:_ [_Medium Membership_](https://adithyask.medium.com/membership)_. Your support is greatly appreciated!_
