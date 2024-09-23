Title: From 7B to 8B Parameters: Understanding Weight Matrix Changes in LLama Transformer Models

URL Source: https://medium.com/@adithyask/from-7b-to-8b-parameters-understanding-weight-matrix-changes-in-llama-transformer-models-31ea7ed5fd88

Published Time: 2024-04-19T08:20:53.578Z

Markdown Content:
[![Image 1: Adithya S K](https://miro.medium.com/v2/resize:fill:88:88/1*w1_VSVDg5oqt19oTB4MAMg.jpeg)](https://adithyask.medium.com/?source=post_page-----31ea7ed5fd88--------------------------------)

Deep Dive into the Underlying Architecture of LLama3

Quick links for Llama3:

> GitHub repo — [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)
> 
> Huggingface model link- [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
> 
> Official website — [https://ai.meta.com/blog/meta-llama-3/](https://ai.meta.com/blog/meta-llama-3/)

Quick notes on the release:

*   8 billion and 70 billion (8B and 70B) contexts
*   Context length up to 8K
*   Knowledge cutoff about a year ago
*   English language only
*   Pretrained on 15 trillion tokens, fine-tuned on 10 million human-annotated instructions (not released)
*   Significantly better than Llama2, which is encouraging for all other open-source software finetunes

**Before going into the differences at the weight matrix, let’s first understand the high-level differences.**
--------------------------------------------------------------------------------------------------------------

![Image 2](https://miro.medium.com/v2/resize:fit:700/1*Kvhy8TFGQNZAvmgRV0x5XA.jpeg)

This tweet by [Andrej Karpathy](https://twitter.com/karpathy/status/1781028605709234613) explains the differences in detail:

If you wondered why LLama3 is 8b parameters instead of 7B (~6.74 B), here are the changes in matrix sizes between LLama3 and LLama2:
------------------------------------------------------------------------------------------------------------------------------------

**Embedding Layer:**

*   LLama3: Dimensions `[128256, 4096]`
*   LLama2: Dimensions `[32000, 4096]`
*   Difference in vocabulary size:
*   LLama3 has a larger vocabulary size compared to LLama2, with 128256 tokens versus 32000 tokens. This difference in vocabulary size leads to a larger embedding matrix in LLama3.

**Output Layer (lm\_head.weight):**

*   Both LLama3 and LLama2 have the same output layer dimensions `[Vocabulary Size, 4096]`, where the vocabulary size is dependent on the tokenization scheme. In this case:
*   LLama3: Vocabulary Size = 128256
*   LLama2: Vocabulary Size = 32000

The increase in vocabulary size in LLama3 necessitates a larger embedding layer compared to LLama2, contributing significantly to the increase in total parameters observed between the two models. This change allows LLama3 to handle a larger and potentially more diverse range of tokens during processing, which can be advantageous for certain natural language processing tasks.

In addition to the vocabulary size difference in the embedding layer, let’s highlight the changes in dimensions for the `mlp.down_proj.weight`, `mlp.gate_proj.weight`, and `mlp.up_proj.weight` matrices between LLama3 and LLama2:

**MLP Down Projection (**`**mlp.down_proj.weight**`**):**

*   LLama3: Dimensions `[4096, 14336]`
*   LLama2: Dimensions `[4096, 11008]`
*   Difference in matrix size:
*   LLama3 has a wider projection matrix compared to LLama2. The number of output features (14336) in LLama3 is higher than that in LLama2 (11008). This change likely enables a more complex transformation within the multi-layer perceptron (MLP) component of each transformer layer in LLama3.

**MLP Gate Projection (**`**mlp.gate_proj.weight**`**):**

*   LLama3: Dimensions `[14336, 4096]`
*   LLama2: Dimensions `[11008, 4096]`
*   Difference in matrix size:
*   LLama3 has a larger input dimension for the gate projection compared to LLama2. The number of input features (14336) in LLama3 is higher than that in LLama2 (11008). This alteration can affect the capacity and expressiveness of the MLP gating mechanism within each transformer layer.

**MLP Up Projection (**`**mlp.up_proj.weight**`**):**

*   LLama3: Dimensions `[14336, 4096]`
*   LLama2: Dimensions `[11008, 4096]`
*   Difference in matrix size:
*   Similar to the gate projection, LLama3 employs a larger input dimension for the up projection compared to LLama2. The number of input features (14336) in LLama3 is greater than that in LLama2 (11008), likely contributing to increased model complexity and capacity in LLama3.

These changes in the dimensions of the projection matrices (`mlp.down_proj.weight`, `mlp.gate_proj.weight`, `mlp.up_proj.weight`) reflect adjustments made to the internal architecture of each transformer layer in LLama3 compared to LLama2. The increase in dimensions allows LLama3 to potentially capture more intricate patterns and dependencies during the processing of input sequences, which can be beneficial for handling diverse and complex language tasks.

Weight Matrix Breakdown and Parameter Calculation
-------------------------------------------------

About 6 to 7 months ago, I always wondered what parameters mean and how they are determined. If you also had questions like this, the explanation below will give you an understanding of how the breakdown and calculation happens.

LLama 3 Weight Matrix breakdown
-------------------------------

**Embedding Layer (**`**model.embed_tokens.weight**`**):**

*   Dimensions: `[128256, 4096]`
*   Total parameters: (128256 \* 4096 = 525336576)

**Each Transformer Layer (**`**model.layers.0**` **to** `**model.layers.31**`**):**

Each layer consists of several weight matrices:

*   `input_layernorm.weight`: `[4096]`
*   `mlp.down_proj.weight`: `[4096, 14336]`
*   `mlp.gate_proj.weight`: `[14336, 4096]`
*   `mlp.up_proj.weight`: `[14336, 4096]`
*   `post_attention_layernorm.weight`: `[4096]`
*   `self_attn.k_proj.weight`: `[1024, 4096]`
*   `self_attn.o_proj.weight`: `[4096, 4096]`
*   `self_attn.q_proj.weight`: `[4096, 4096]`
*   `self_attn.v_proj.weight`: `[1024, 4096]`

Total parameters for each layer:

*   `input_layernorm.weight`: (4096)
*   `mlp.down_proj.weight`: (4096 \* 14336 = 58720256)
*   `mlp.gate_proj.weight`: (14336 \* 4096 = 58720256)
*   `mlp.up_proj.weight`: (14336 \* 4096 = 58720256)
*   `post_attention_layernorm.weight`: (4096)
*   `self_attn.k_proj.weight`: (1024 \* 4096 = 4194304)
*   `self_attn.o_proj.weight`: (4096 \* 4096 = 16777216)
*   `self_attn.q_proj.weight`: (4096 \* 4096 = 16777216)
*   `self_attn.v_proj.weight`: (1024 \* 4096 = 4194304)
*   Total parameters per layer: \[4096 + 58720256 + 58720256 + 58720256+ 4096 + 4194304 + 16777216 + 16777216 + 4194304 = 218112000\]

**Output Layer (**`**lm_head.weight**`**):**

*   Dimensions: `[128256, 4096]`
*   Total parameters: (128256 \* 4096 = 525336576)

**Total Parameters for 32 Layers:**

*   Total parameters per layer: (218112000)
*   Total parameters for 32 layers: (32 \* 218112000 = 6979584000)

Input embedding Layer + Output layer = 1050673152

> Overall parameters = Total Parameters for 32 Layers + Input and Output Layers = 8030257152 (8.03 Billion Parameters)

Therefore, the total number of parameters in this transformer architecture with 32 layers is **8,030,257,152 (8.03 B)** parameters.

Llama 2 Weight Matrix breakdown
-------------------------------

To calculate the number of parameters for the given transformer architecture, we need to consider the dimensions of each weight matrix and count the total number of elements (parameters) across all layers.

Let’s break down the calculation step by step:

**Embedding Layer (**`**model.embed_tokens.weight**`**):**

*   Dimensions: `[32000, 4096]`
*   Total parameters: (32000 \* 4096 = 131072000)

**Each Transformer Layer (**`**model.layers.0**`**):**

Each layer consists of several weight matrices:

*   `input_layernorm.weight`: `[4096]`
*   `mlp.down_proj.weight`: `[4096, 11008]`
*   `mlp.gate_proj.weight`: `[11008, 4096]`
*   `mlp.up_proj.weight`: `[11008, 4096]`
*   `post_attention_layernorm.weight`: `[4096]`
*   `self_attn.k_proj.weight`: `[4096, 4096]`
*   `self_attn.o_proj.weight`: `[4096, 4096]`
*   `self_attn.q_proj.weight`: `[4096, 4096]`
*   `self_attn.v_proj.weight`: `[4096, 4096]`
*   `self_attn.rotary_emb.inv_freq`: `[64]` (not counted as parameters in this context)

Total parameters for each layer:

*   `input_layernorm.weight`: (4096)
*   `mlp.down_proj.weight`: (4096 \* 11008 = 45088768)
*   `mlp.gate_proj.weight`: (11008 \* 4096 = 45088768)
*   `mlp.up_proj.weight`: (11008 \* 4096 = 45088768)
*   `post_attention_layernorm.weight`: (4096)
*   `self_attn.k_proj.weight`: (4096 \* 4096 = 16777216)
*   `self_attn.o_proj.weight`: (4096 \* 4096 = 16777216)
*   `self_attn.q_proj.weight`: (4096 \* 4096 = 16777216)
*   `self_attn.v_proj.weight`: (4096 \* 4096 = 16777216)
*   Total parameters per layer: \[4096 + 45088768+ 45088768+ 45088768+ 4096 + 16777216+ 16777216+ 16777216+ 16777216\] = 202383360

**Output Layer (**`**lm_head.weight**`**):**

*   Dimensions: `[32000, 4096]`
*   Total parameters: (32000 \* 4096 = 131072000)

**Total Parameters for the Layer:**

*   Total parameters per layer: 202383360
*   Total parameters for the 32 layer: 32\*202383360 = 6476267520

Input embedding Layer + Output layer = 262144000

> Overall parameters = Total Parameters for 32 Layers + Input and Output Layers = 6738411520(6.74 Billion Parameters)

Therefore, the total number of parameters in this LLama2–7b is **6738411520(6.74 B)** parameters.

This calculation includes all weight matrices within the specified layer. Note that the calculation does not include the `self_attn.rotary_emb.inv_freq` parameter as it's a single-dimensional vector and is typically not considered as part of the parameter count for the layer.

Closing Thoughts
----------------

*   Firstly, I was a little sad because there hasn’t been much architectural change, and adding “Llama3” before the name of every finetune seems a bit overboard.
*   From my testing, the model is really capable across a lot of tasks, and it’s quite robust. It’s not as guardrailed as LLama2.
*   Regarding the Indic LLM landscape, it will be very challenging to finetune it for Indic languages because the tokenizer isn’t very efficient. From my testing, it’s only better at tokenizing Devanagari-based text than LLama2. However, for other languages like Kannada, Tamil, Telugu, Malayalam, etc., it’s going to be tough as we’ll have to expand the vocabulary, which will require a lot of continual pretraining, considering this was trained on 15T tokens.

here is a tweet that goes over “why it will be diffcult to finetune Llama3 for indic langauges”

> If you found this post valuable, make sure to follow me for more insightful content. I frequently write about the practical applications of Generative AI, LLMs, Stable Diffusion, and explore the broader impacts of AI on society.
> 
> Let’s stay connected on [Twitter](https://twitter.com/adithya_s_k). I’d love to engage in discussions with you.
> 
> If you’re not a Medium member yet and wish to support writers like me, consider signing up through my referral link: [Medium Membership](https://adithyask.medium.com/membership). Your support is greatly appreciated!