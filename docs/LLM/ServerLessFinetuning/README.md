# Write code locally and run it on GPUs in Seconds using Modal

## Stop Paying for Idle GPUs: Serverless Training with Modal

So let's face it, if you are doing anything with deep learning, GPUs are a must.

They are expensive, and infrastructure is hard to set up. Most of the time, you code and stuff when the GPUs are sitting idle, and it's a pain to pay for the uptime when it's literally sitting idle, and no deep learning scripts work on the first go.

This was a problem I faced as "GPU Poor". I didn't want to spend money on GPUs when I was coding or doing something that didn't leverage the GPU compute. Even for things like downloading data, models, and data transformation, you don't need a GPU but still have to do it on a GPU.

And especially on cloud providers, where you will have to worry about infrastructure. You can set up a VM with a GPU attached, then choose an image which is not even well-documented. If not done properly, you will have to install CUDA and stuff from scratch. Then install it; if that also doesn't work, most of the time you run into a Docker container with the right installations.

And if you start doing multi-GPU training, that's one more burden. Some GPU images don't even support NCCL for communication between GPU nodes, so you will have to be careful about that as well.

So if you just want to set up a GPU and run, it's a lot of effort. There are providers like Runpod, Vast AI, and others that make it easier.

I run a research lab called CognitiveLab [cognitivelab.in], where we do a bunch of model training, synthetic data generation, RL runs, and more. We wanted something that was easy to use, train, and flexible enough so that we don't need to be constrained by it.

But when I looked for a solution where I could **write my code locally on my machine and run it on a GPU**, I stumbled across this beautiful solution called [Modal](https://modal.com). It's been 1 year since I started using it, and it's been a blessing.

I want to share how I exactly use Modal to write and experiment with training scripts locally and run them on GPUs as quickly as possible.

I will cover how to handle datasets on Modal, how to write the training run (using libraries like Unsloth, Axolotl, MS Swift), how to then evaluate these models, and finally how to serve them in a scalable manner using vLLM.

> I will be mainly covering SFT examples, but if you guys are interested, I will write a blog on how to set it up for RL with RL training and reward environments happening on different GPUs.

## Inspiration

[**Thinking Machines**](https://x.com/thinkymachines), the startup from ex-OpenAI CTO Mira Murati, recently launched Tinker with the ability to write training loops in Python on your laptop; we'll run them on distributed GPUs.

https://x.com/thinkymachines/status/1973447428977336578

Which is every developer's dream, but I had been using Modal to do the something similar for a while now.

> **PS:** From the looks of it, their API is much more sophisticated, they have done a lot of optimisations under the hood using batching efficiently here is a tweet that goes more into details [tweet link](https://x.com/cHHillee/status/1973469947889422539)

I thought I would write this blog to share how I have been able to do something similar using something called Modal (I love Modal!!!).

## Ok, what is Modal?

You would have come across the term Serverless GPUs.

Let's just say Modal is a GPU provider platform that does right by serverless GPUs, and they have one of the best developer experiences ever.

If you are doing anything in Python, training models, deploying them, writing servers, building agentic systems, then Modal can be used.

As per the official Modal website:

> AI infrastructure that developers love, and that's 100% factual.
>
> Run inference, train, batch process with sub-second cold start, instant auto-scaling, and a developer experience that feels local.

**Fun fact:** Even Lovable uses Modal for running their sandbox.

### Getting Started

So let's get started. First, all you have to do is:

```bash
pip install modal
```

and do:

```bash
modal setup
```

You can also authenticate through their API keys:

```bash
export MODAL_TOKEN_ID=
export MODAL_TOKEN_SECRET=
```

This is all you need to set up Modal.

### Core Concepts

With Modal, you always start by creating an App, an Image, and Volumes.

**App** - So to set an App, it's pretty simple:

```python
import modal

# Create the Modal app
app = modal.App("<app_name>")
```

**Volumes** - Then we can create or use existing volumes.

You can think of volumes as a storage file system where you can store anything like model weights, datasets, scores, and more.

If you want something to persist, add it in the volume. The best part is, for a function, you can have multiple volumes. Add different routes; you can have a volume that's for model weight in the `/model` path and for the dataset in the `/dataset` path.

Something like this:

```python
dataset_volume = modal.Volume.from_name("dataset-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("model-volume", create_if_missing=True)
```

Then you write the mapping that will be passed into functions:

```python
volume_config = {
    "/dataset": dataset_volume,
    "/model": model_volume
}
```

This is just to illustrate how you can attach volumes to any function. This gives us awesome power.

You can download datasets, process them all on CPU instances, and when it comes time to train, just attach the same volume and use it, which makes life that easy.

> I generally create a volume for a single experiment or training run so that I have everything consolidated that can be used across the project.

**Images** - Next thing will be the images.

This is the most important part. I would say defining an image can be tricky at first, but once it's done, you don't have to worry about it. Initially, it can take up some time.

But it's very important. I would say refer to [Modal Image docs](https://modal.com/docs/reference/modal.Image) to see all the ways to create an image.

Here is a sample example image:

```python
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
        "wandb==0.21.0",
    )
    .env({"HF_HOME": "/model_cache"})
)
```

So the base image uses Debian with Python 3.11, and then you install all the packages using uv. Then you set the environment `HF_HOME` so that everything is cached, and you won't have to download again and again. This is a good starting image.

> **Pro tip:** At the end of this blog, I will give you a set of images that you can use for anything training-related that I have used. I have images to serve LLM using vLLM, SGLang, training using Unsloth, MS Swift, and more. I will go deep into how to create images in a better way at the end of the blog.

**Functions** - There is one more thing: Functions.

You can basically have any Python function to make it run on GPU or CPU on Modal. All you have to do is add a decorator:

```python
@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],  # local .env variables
    volumes={"/data": volume},
    timeout=3600,  # 1 hour timeout
)
def any_python_function():
    # Your code here
    pass
```

And it's very important.

Here is where you define which image the function will use, what secrets you will be passing, which volumes will be attached, and what is the timeout (there is a general timeout of 24 hrs).

To know all the properties of a function, refer to [Modal Function docs](https://modal.com/docs/reference/modal.Function)

---

Now the basics are out of the way. Let's do some training, fine-tuning, evaluation, and serving.


Let's get started.

## Tutorials

I have created comprehensive tutorials for each training approach:

### 1. [Training NanoGPT on Modal](TrainNanoGPTModal.md)


📄 **[View Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/TrainNanoGPTModal.py)**

Learn how to take an existing codebase (Andrej Karpathy's nanoGPT) and run it on Modal's serverless GPUs with minimal modifications. Perfect for beginners to understand:

- How to copy local repositories into Modal containers
- Data preparation, training, and sampling pipelines
- Managing persistent storage with Modal volumes
- Running existing Python projects on remote GPUs

| **Level** | **GPU Required** | **Time** |
|-----------|------------------|----------|
| Beginner | 1× A100-40GB (or T4/L40S for testing) | 30 mins - 2 hours |

---

### 2. [Fine-tuning Gemma 3-4B with Unsloth](FinetuneGemmaUnslothModal.md)

**End-to-end vision-language model training and deployment**

📄 **[View Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/FinetuneGemmaUnslothModal.py)**

A production-grade pipeline covering the complete ML workflow from data to deployment. You'll learn:

- Fine-tuning vision-language models with LoRA
- Optimized single-GPU training with Unsloth
- Model evaluation with automated metrics
- Serving with vLLM for high-throughput inference
- Auto-scaling deployment strategies

| **Level** | **GPU Required** | **Time** |
|-----------|------------------|----------|
| Intermediate | 1× A100-80GB (or L40S) | 3-6 hours (full pipeline) |

---

### 3. [Multi-GPU Training with Axolotl](FinetuneLlamaAxolotlGPUModal.md)

**Distributed training for large models (Llama 8 - 70B+)**

📄 **[View Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/FinetuneLlamaAxolotlGPUModal.py)**

Advanced distributed training techniques for massive models. You'll learn:

- Multi-GPU training with Accelerate and DeepSpeed
- YAML-based configuration for reproducibility
- Dataset preprocessing for large-scale training
- Scaling from 2 to 8 GPUs
- Cost optimization strategies for expensive training runs

| **Level** | **GPU Required** | **Time** |
|-----------|------------------|----------|
| Advanced | 2-8× A100-80GB | 4-12 hours (depends on model size) |

> This is multi-GPU training, and you can run all types of parallelism (data, tensor, pipeline, FSDP) using Modal as well. For the sake of simplicity, I have used Accelerate, but you can go all out with the setup up to 8 GPUs. I have mainly been using Modal for multi-GPU training with a maximum of 8 GPUs. I haven't done multi-node training yet (should be possible with sandboxes, but the setup process might be a bit complex).

---

I think these 3 examples will give you a good picture to replicate the process across multiple things.

## Final thoughts

As someone working with AI models, infrastructure is crucial to get right as its expensive and take a lot of time to set up , especially individual researcher and small lab will find it hard to set up and manage infrastructure.

With Modal, infrastructure becomes as easy as writing a python script and running it on CPU GPU deploying it scaling it.

In this I go over the details on how to use modal mainly for running training eval and serving scripts for LLM models but you can do a lot more with modal.

> **Fun fact:** [Gitvizz](https://gitvizz.com) uses modal to run all their backend code and I have been running it for 4 months which cost me less than 4$ and it scales really well. After using modal I completely stopped using k8s and stuff for smaller projects.

---

### Need Help?

If your organization needs help with optimally using modal we at [CognitiveLab](https://cognitivelab.in) can help you set it up and manage it for you.

Reach out to us through our website or DM me on twitter [@adithya_s_k](https://x.com/adithya_s_k)

## Resources

- [Modal Docs](https://modal.com/docs)
- [Unsloth Docs](https://docs.unsloth.ai/)
- [Axolotl Docs](https://docs.axolotl.ai/)
- [NanoGPT](https://github.com/karpathy/nanoGPT)