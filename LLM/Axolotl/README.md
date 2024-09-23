## Introduction

Welcome to an immersive journey into the intricate art of fine-tuning AI models, where precision intertwines with customization. In this blog, we will delve into the profound significance of fine-tuning, unravel its complexities, and unveil a remarkable tool that streamlines this process - Axolotl.

Get ready to master the art of fine-tuning an LLM (Large Language Model) using Axolotl. Throughout this tutorial, we'll navigate through the intricacies and adjustable elements of this transformative process.

Axolotl predominantly operates through YAML files, requiring you to engage with and customize these files as part of the workflow.

The Axolotl repository boasts comprehensive documentation, ensuring a swift grasp of its functionalities to kickstart your journey seamlessly.

Join me as we unravel the journey from ideation to fine-tuning an LLM, leveraging Axolotl as our medium of choice for this transformative endeavor. Let's dive in!

Introducing the Axolotl UI Editor

Before we proceed further, I'd like to introduce the Axolotl UI Editor—a powerful tool designed to facilitate the editing of Axolotl YAML files for fine-tuning. This intuitive editor is a handy creation from [CognitiveLab](https://www.cognitivelab.in/), providing a seamless interface for customizing Axolotl configurations.

Explore the Axolotl UI Editor here: [Axolotl UI Editor](https://axolotl-ui.vercel.app/)

!https://prod-files-secure.s3.us-west-2.amazonaws.com/97859362-02bd-4d74-abc1-b66ffcf4d0ad/6ecd3ef8-c16b-4e2f-bb05-58503b49e969/Untitled.png

You can also find the source code for this editor on GitHub: [Axolotl UI Editor - GitHub](https://github.com/adithya-s-k/axolotl-ui)

This user-friendly interface simplifies the process of tweaking Axolotl YAML files, making it accessible and efficient for fine-tuning tasks. Now, let's dive deeper into leveraging this tool for our fine-tuning journey.

## What is Fine Tuning?

Before we delve into Axolotl, let's understand the significance of fine-tuning in the world of artificial intelligence. Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task or dataset. It's akin to honing a skill - refining an already proficient model for a specialized purpose.

Some examples of fine-tuning tasks can include:

- Fine-tuning for generating structured output (e.g., function calling).
- Fine-tuning to emulate someone's style or behavior.

## Introduction to Axolotl

Axolotl, a versatile tool designed for AI model fine-tuning, emerges as a game-changer in the field. It supports various Hugging Face models, including llama, pythia, falcon, and mpt, empowering users with a multitude of configurations and architectures.

### **Key Features:**

1. **Model Support:** Easily train models such as llama, pythia, falcon, and mpt.
2. **Configurability:** Effortlessly customize configurations using a simple YAML file or CLI overwrite.
3. **Dataset Flexibility:** Load datasets in various formats, use custom formats, or bring your own tokenized datasets.
4. **Advanced Techniques:** Benefit from integrated features like xformer, flash attention, rope scaling, and multipacking.
5. **Scalability:** Run on a single GPU or multiple GPUs using FSDP or Deepspeed.
6. **Containerization:** Seamlessly run with Docker, either locally or in the cloud.
7. **Logging and Checkpoints:** Log results and optionally save checkpoints to wandb for comprehensive tracking.

Axolotl provides a comprehensive suite of tools and capabilities, making AI model fine-tuning accessible, efficient, and adaptable to diverse use cases. Let's explore how to harness these features for optimizing and refining our models effectively.

## Axolotl supports the following models

|  | fp16/fp32 | lora | qlora | gptq | gptq w/flash attn | flash attn | xformers attn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mistral | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mixtral-MoE | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ |
| Pythia | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❓ |
| cerebras | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❓ |
| btlm | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❓ |
| mpt | ✅ | ❌ | ❓ | ❌ | ❌ | ❌ | ❓ |
| falcon | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❓ |
| gpt-j | ✅ | ✅ | ✅ | ❌ | ❌ | ❓ | ❓ |
| XGen | ✅ | ❓ | ✅ | ❓ | ❓ | ❓ | ✅ |
| phi | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ |
| RWKV | ✅ | ❓ | ❓ | ❓ | ❓ | ❓ | ❓ |
| Qwen | ✅ | ✅ | ✅ | ❓ | ❓ | ❓ | ❓ |

## **Idea 1: YouTube Cloner**

Have you ever wondered if you could clone a YouTuber? I recently had this intriguing idea to explore whether a language model could mimic the style of popular YouTubers. The concept is simple yet fascinating: using fine-tuning techniques with LoRA, I aim to teach a language model how to emulate the tone, pacing, and content style of specific YouTube channels.

### **Project Objective**

The aim of the YouTube Cloner project is to investigate the abilities of Language Models (LLMs) in mimicking the speaking style of popular YouTubers. Its main goal is to determine the effectiveness of LLMs in replicating the tone, pacing, and content style of specific YouTube channels through fine-tuning on selected datasets.

Objectives

- **Explore Emulation:** Assess the ability of LLMs to mirror the unique tone and pacing of popular YouTube channels.
- **Fine-Tuning Experiment:** Implement fine-tuning methods with hand-picked datasets to train LLMs for channel-specific styles.

Join me on this exciting journey as we unravel the potential of language models in mimicking the essence of favorite YouTube personalities. Let's dive into the details and see how we can bring this idea to life!

### **Dataset Preparation**

Before diving into the fine-tuning process, we need to prepare the dataset that will enable us to achieve our goal.

If you're interested in recreating this project, I've open-sourced the web scraping and dataset curation code [here](https://github.com/adithya-s-k/LLM-Alchemy-Chamber/tree/main/Projects/YT_Clones). Feel free to star the repository if you find it helpful!

I've also made the dataset available [here](https://huggingface.co/datasets/AdithyaSK/Fireship_transcript_summar_prompt) on Hugging Face Datasets. You can explore the notebooks containing the code or directly use the dataset to replicate the steps I followed.

[](https://github.com/adithya-s-k/LLM-Alchemy-Chamber/tree/main/Projects/YT_Clones)

[AdithyaSK/Fireship_transcript_summar_prompt · Datasets at Hugging Face](https://huggingface.co/datasets/AdithyaSK/Fireship_transcript_summar_prompt)

### Tips on Dataset Curation

Dataset curation is a critical step often overlooked in many guides or tutorials, but I'll delve into the details behind building a good fine-tuning dataset.

Firstly, you need to decide your input prompt and what you expect from the model. In the case of the YouTube Cloner, I aimed to use the video title and a brief summary as input to generate the video script.

During dataset creation, my primary focus was to gather the YouTube video titles and transcripts and generate concise summaries. Here's how I accomplished this:

1. Extracted links from a YouTube channel through web scraping.
2. Downloaded audio, titles, and other relevant information.
3. Transcribed the audio into text using a deep-seed audio-to-text API and generated summaries.

Now, we have all the necessary components: titles, summaries, and video transcripts. Next, it's time to format the prompt.

When fine-tuning base models like llama or mistal that are not instructionally fine-tuned, you can use your own prompt format. However, if the model is instructionally fine-tuned, you must follow the instruction format.

Axolotl supports various dataset types like alpaca, llama, etc., but I generally prefer **`completion`** as it offers the most control over dataset formatting.

Here is the prompt format I chose for this particular project:

```
<s>
[INST]
You are a YouTuber called Fireship, creating engaging, high-intensity coding tutorials and tech news.
You cover a wide range of topics relevant to programmers, aiming to help them learn and improve their skills quickly.

Given the title of the video: {title}
and a small summary: {video_summary}
[/INST]

Generate the video: {video_transcipt}
</s>
```

Here, "<s>" is the start-of-sentence token and "</s>" is the end-of-sentence token. These are unique tokens.

The "</s>" token indicates to the generator when to stop generating.

So, essentially, these are important aspects to consider while fine-tuning.

Providing these as the input is crucial.

```
<s>[INST]
You are youtuber called Fireship you make engaging high-intensity and entertaining coding tutorials and tech news.
you covers a wide range of topics relevant to programmers, aiming to help them learn and improve their skills quickly.

Given the title of the video : Rust in 100 seconds
and a small summary : A 100 second video on Rust not a code report
[/INST]
```

With the title and summary,

And you are making the LLM familiar with giving a response like this.

```
Generate the video : Rust. A fast and memory efficient language known for taking everything that's wrong with low level systems programming languages like C plus plus, c, and assembly, then making it worse by eliminating pointers entirely and providing an unpronounceable name that makes developers angry. It was created by Graydon Hoare inspired by ML functional languages and aimed at building safe reliable software while remaining extremely fast. In fact, its motto is secure concurrency without sacrificing speed, which sounds almost too good to be true. The main problem with unsafe multi threaded programming today is data races. Data races happen when multiple threads try to access the same mutable value at the exact same time, but there's no way to know which thread will win, making your programs unreliable. It can also cause hard to debug errors because you have race conditions where each one occurs randomly based on timing issues between different threads. There are many libraries out there designed to mitigate this issue or prevent it altogether. But, rust takes a very different approach by simply disallowing shared state, requiring all values, including variables and functions to either be immutable or only changed via a unique owner called a reference type or pointer. This means data races cannot occur automatically in a typical application, though they can still be introduced accidentally if someone tries to use the borrow checker to perform some fancy hackery. What that usually results in is clear error messages about double usage of references instead of cryptic seg faults and other run time errors, allowing you to solve problems earlier in development, and with less frustration. And it allows you to do so with patterns already proven to work reliably. For example, smart pointers eliminate the need for things like raw pointers and free store allocators found in C plus plus, ensuring proper garbage collection. Instead of using inheritance, interfaces, generics, traits provide object oriented features such as polymorphism in a statically typed way. As awesome as that sounds, learning curves don't come much steeper than rust, mostly because of its ownership system, which I would describe as more of a philosophy than anything else. If you want the full explanation, subscribe to my channel. Otherwise, just enjoy these 2 hello worlds from the same file. You might think the first line here declares a variable named hello with the string hello world assigned to it. However, you'd be wrong. That doesn't actually define a new variable. Rather, It defines a function with an explicit return type of a string literal. When used in conjunction with println, it prints the string literally twice. Or we could define a global variable with mut, which changes the meaning of the assignment operator to mutate existing memory. Now, let me read you something really scary. To get rid of pointers completely. We have references instead. These act exactly like the address of operators in other languages, except they implement safety checks through rust's highly sophisticated borrow checker. On top of that, you can clone objects into new locations, move values around, deep copy and shallow copy across types, weak references, arc, ref cell, interior, pin, once cell, and on and on. At this point, you should start seeing how rust got its name. If you wanna build a complex multi threaded system with performance requirements. Your best bet may well be learning this crazy language that seems so easy on the surface. This has been the rust programming language in 100 seconds. Hit the like button if you wanna see more short videos like this. Thanks for watching and I will see you in the next one.
</s>
```

so this will be the main dataset prep part

After formatting the dataset, store all the formatted prompts in a field or column named `text`. This step is crucial before pushing it to Hugging Face.

[Share a dataset to the Hub](https://huggingface.co/docs/datasets/v2.17.1/en/upload_dataset#upload-with-python)

### Fine-tuning

After you have uploaded the dataset, 90% of the work is complete. All that's left to do now is use Axolotl to fine-tune the model.

Axolotl makes it really easy. All you have to do is determine which type of fine-tuning you want to do, be it Lora, FFT, or qLora.

Here is a high-level abstraction to help you decide which one you want to choose:

Lora - If you want to fine-tune the model to respond in a particular type - like JSON, like the above YouTube cloner example. Domain adaptation using Lora is tough. Either you have to have a very high rank value to change a lot of parameters, or you will have to overfit on the data which might degrade your model's performance.

If you want to fine-tune a model to respond to you in a particular style, then choose Lora. Generally, I fine-tune most of my models using Lora.

qLora - If you want to perform Lora but don't have enough compute, you can use qLora which lets you fine-tune models on a lower-end GPU like T4 with 16GB of VRAM, but the time for fine-tuning will increase.

FFT - Full weight fine-tuning - in Axolotl you can FFT llama on an A100-80GB variant because it is well optimized. I generally go with FFT for domain or language adaptation, but FFT might decrease the original model's performance.

Now, while performing Lora, there are some more things you have to note down.

The Rank - R and alpha values.

Here is a tweet I wrote about the considerations to make to pick an R and alpha value:

https://publish.twitter.com/?url=https://twitter.com/adithya_s_k/status/1744065797268656579#

Now that that's out of the way, let's get into fine-tuning the model.

# **Prerequisites (Optional)**

There are two primary prerequisites: a server with an Ampere GPU, such as A100, and a functioning conda setup.

For the A100 GPU, you can utilize any cloud platform such as Google Cloud, AWS, Lambda Labs, or E2E Networks. Your choice may depend on your region and budget.

local install

```bash
git clone <https://github.com/OpenAccess-AI-Collective/axolotl>
cd axolotl

pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'

```

docker 

```bash
  docker run --gpus '"all"' --rm -it winglian/axolotl:main-py3.10-cu118-2.0.1

```

assuming you have set up axolotl properly and have everything properly configured 

Pick a model you want to finetuned 

in this case i wanted to using the [base mistral model](https://huggingface.co/mistralai/Mistral-7B-v0.1)  

you can see a folder called [examples](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples) and you can see all the models axolotl supports and their finetuning scripts

now lets create a copy of the qLora finetunign yml and call it lora.yml and make the necessary changes

[](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/qlora.yml)

as we will be using Lora and a different dataset and different rank

here is the `yml` script to finetune the model 

```yaml
base_model: mistralai/Mistral-7B-v0.1 
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizer
is_mistral_derived_model: true

load_in_8bit: false
# change load_in_4bit to false as we will be doing lora finetuning
load_in_4bit: false # change from true to false
strict: false

datasets:
  - path: AdithyaSK/Fireship_transcript_summar_prompt # change the dataset to the dataset you have pushed on HF
    type: completion #change it from alpace to completion
dataset_prepared_path: last_run_prepared
val_set_size: 0.1
output_dir: ./lora-out # change the directory name - it will the folder in which the final model will be stored

adapter: lora
lora_model_dir:

sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true

lora_r: 64 # lets change the rank from 32 to 64 to change more parameters in the model
lora_alpha: 32 # lets change alpha from 16 to 31 general rule of thumb is r = 2*aplha
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:
lora_target_modules: 
  - gate_proj
  - down_proj
  - up_proj
  - q_proj
  - v_proj
  - k_proj
  - o_proj

wandb_project: # if you want to track the models 
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 4 # gradient accumulation can be increase to 8 but lets it keep it at 4. if you are getting cuda_out_of_memoery you can reduce this number
micro_batch_size: 2 
num_epochs: 5 # increase the number of epochs to 5. 1 epoch = the model has seen all the data once
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

loss_watchdog_threshold: 5.0
loss_watchdog_patience: 3

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
eval_max_new_tokens: 128
saves_per_epoch: 4 # change it from 1 to 4 to to make multiple save during the Epoch
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens: # very important 
  bos_token: "<s>" # these tokens will be added at the first of every prompt
  eos_token: "</s>" # these tokens will be added at the end of every prompt
  unk_token: "<unk>"
```

after you have made the necessary changes to the finetuning script

to run the finetuning 

```yaml
accelerate launch -m axolotl.cli.train examples/mistral/lora.yml
```

to run inference on the model and test out the model 

```yaml
python -m axolotl.cli.inference examples/mistral/lora.yml --lora_model_dir="./lora-out"
```

to merge the model

```yaml
python3 -m axolotl.cli.merge_lora examples/mistral/lora.yml --lora_model_dir="./lora-out"
```

push the model to huggingface 

```yaml
pip install -U "huggingface_hub[cli]"
huggingface-cli upload {name of model} ./{output directory} --repo-type model
```

### Conclusion

In this blog post, we've explored the exciting world of fine-tuning AI models using Axolotl, a powerful tool that enhances the customization and precision of language models. We've discussed the importance of dataset preparation, model selection, and configuration settings for successful fine-tuning projects.

By leveraging Axolotl's capabilities, we've embarked on intriguing projects like the YouTube Cloner, aiming to emulate the speaking style of popular YouTubers using language models. Throughout this journey, we've emphasized the creative potential and experimental nature of fine-tuning, showcasing how AI can adapt to specific tasks and datasets.

Now equipped with the necessary knowledge and tools, you're ready to embark on your own fine-tuning adventures. Whether it's replicating existing projects or exploring new use cases, Axolotl offers a robust framework to streamline and optimize the fine-tuning process.

## Closing Thoughts

As we conclude this exploration into fine-tuning using Axolotl, it's evident that the tool not only simplifies the process but also enhances the capabilities of AI practitioners. The ability to fine-tune models with precision, coupled with the flexibility offered by Axolotl, opens doors to a new era of AI customization.

Embark on your journey with Axolotl, where every fine-tuning endeavor transforms into a seamless and efficient experience. Stay tuned for more insights and updates as we navigate the evolving landscape of AI refinement.

> *If you found this post valuable, make sure to follow me for more insightful content. I frequently write about the practical applications of Generative AI, LLMs, Stable Diffusion, and explore the broader impacts of AI on society.*
> 

> *Let's stay connected on [Twitter](https://twitter.com/adithya_s_k). I'd love to engage in discussions with you.*
> 

> *If you're not a Medium member yet and wish to support writers like me, consider signing up through my referral link: [Medium Membership](https://adithyask.medium.com/membership). Your support is greatly appreciated!*
> 

### Resources

[Fine-tuning Llama 2 with axolotl](https://dzlab.github.io/dltips/en/pytorch/llama-2-finetuning-axolotl/)

[A Beginner’s Guide to LLM Fine-Tuning](https://towardsdatascience.com/a-beginners-guide-to-llm-fine-tuning-4bae7d4da672)

[Finetuning LLMs For Text-to-Task](https://medium.com/@rohanbalkondekar/finetuning-llms-for-text-to-task-c04374454ac0)

[A poor man's guide to fine-tuning Llama 2](https://duarteocarmo.com/blog/fine-tune-llama-2-telegram)

[Fine Tuning Llama Models With Qlora and Axolotl | ANIMAL-MACHINE](https://www.animal-machine.com/posts/fine-tuning-llama-models-with-qlora-and-axolotl/)

https://github.com/OpenAccess-AI-Collective/axolotl