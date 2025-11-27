# Tiny Recursive Models (TRM): When Small Models Think Better

So here's something that caught my attention recently. A 7 million parameter model is absolutely crushing models that are 100,000 times its size on reasoning tasks. I'm talking about 87.4% accuracy on hard Sudoku puzzles while GPT-4, Claude, and even the massive 671B parameter DeepSeek R1 score 0%. Zero. Nada.

And honestly? This isn't about making models bigger. We've been doing that for years. This is about making them think differently.

I spent some time diving deep into this paper and created visualizations to help explain what's going on. By the end of this, you'll see why sometimes the smartest solution isn't more parameters - it's a smarter architecture.

---

## What is TRM?

TRM (Tiny Recursive Models) is a research breakthrough from Samsung SAIL Montreal. It's designed for structured reasoning tasks - Sudoku puzzles, mazes, abstract reasoning problems (ARC-AGI). The core idea is simple but powerful: instead of processing a problem once with a massive network, TRM processes it many times with a tiny network.

Think about how you actually solve a Sudoku puzzle. You don't stare at it for a minute and then write down all 81 numbers in one go. That'd be insane. You fill in some numbers, check your work, realize you made a mistake, backtrack, try again, and slowly iterate toward the solution. That's exactly what TRM does - it iterates and refines.

![type:video](assets/videos/Scene1_Title.mp4)

The results speak for themselves:
- 87.4% on Sudoku-Extreme (vs 55% for its predecessor HRM, 0% for standard LLMs)
- 44.6% on ARC-AGI-1 (a hard abstract reasoning benchmark)
- Only 7M parameters - you could run this on a laptop

---

## The Problem Domain: Why Sudoku?

Let's talk about Sudoku for a minute. It's actually a perfect test case for TRM because it needs both local reasoning (can I put a 5 in this cell?) and global reasoning (does this mess up the entire grid?). But there's something deeper going on here.

Sudoku represents a class of problems where LLMs just... fail. Completely.

Here's why: traditional LLMs process input once and spit out an answer autoregressively - token by token, no going back. For a Sudoku puzzle, this means the model needs to predict all 81 cells in sequence. One wrong prediction early on? The whole thing falls apart. It's like building a house of cards - one mistake and it all collapses.

This is why GPT-4, Claude, and even that monster 671B parameter DeepSeek R1 all score 0% on hard Sudoku. They literally cannot iterate and refine their thinking. They get one shot, and if they mess up cell 12, well, cells 13-81 are probably going to be wrong too.

![type:video](assets/videos/Scene2_SudokuSetup.mp4)

The visualization shows the transformation from problem to solution:

**First**, you see an input puzzle on the left (with dashes for empty cells) and the solved version on the right. Notice how the puzzle is partially filled - about 25-30 cells are given, and the model must deduce the remaining 50+ cells while maintaining consistency across all rows, columns, and 3×3 boxes.

**Second**, watch how the 9×9 grid gets flattened into a sequence of tokens. This is crucial: machine learning models process sequences, not 2D grids. Each cell becomes a single token, reading left-to-right, top-to-bottom. The 9×9 grid becomes a sequence of 81 tokens.

Now here's a technical detail that's actually pretty clever. The vocabulary TRM uses isn't just 0-9 for the digits. It's:
- Token 0: Padding (for smaller grids)
- Token 1: End-of-sequence marker (marks grid boundaries)
- Tokens 2-11: The actual cell values (offset by 2, so token 2 = digit 0, token 11 = digit 9)

Why the offset? Because padding and special tokens need their own IDs. It's a bit awkward but it works. This scheme lets TRM handle any grid size up to 30×30, which is crucial for ARC-AGI tasks where grids vary wildly in size.

And here's the cool part - this same approach works for way more than Sudoku. Mazes? Grid of walls and paths. ARC-AGI puzzles? Grid of colored cells. Game boards? You get it. If you can represent it as a grid, TRM can learn to reason about it.

---

## How Data Flows Through TRM

Alright, before we get to the really clever stuff, let's trace how data actually moves through the model. This is where TRM starts to look different from your standard transformer.

![type:video](assets/videos/Scene3_DataFlow.mp4)

Follow the pipeline from left to right:

**1. Input tokenization**: The 81 cells of a Sudoku grid become token IDs (0-11 as I mentioned earlier). Each token gets embedded into a 512-dimensional vector. Why 512? It's a sweet spot - big enough to capture complex patterns, small enough to not explode the parameter count.

**2. Puzzle ID embedding**: This is clever. When you're training on ARC-AGI, you might see 3 different examples that all follow the same rule (like "rotate 90 degrees then flip colors"). The puzzle ID is a learnable embedding that basically tells the model "hey, these examples are related." Without this, the model would treat each example as completely independent. With it, the model can learn "oh, this is another example of THAT pattern."

**3. Sequence length calculation**: For a 9×9 Sudoku:
- 81 grid tokens
- 1 puzzle ID token
- 15 padding tokens (to reach a round number)
- Total: 97 tokens

For 30×30 grids (ARC-AGI max), the sequence can be up to 900+ tokens. The model uses positional embeddings so it knows which token represents which spatial location.

**4. Dual latent states**: Here's where TRM diverges from standard architectures. The model doesn't just process the input once. It maintains two separate "thought streams":
- **z_H** (High-level state): The current hypothesis for the answer. Think of this as the model's draft solution.
- **z_L** (Low-level state): Reasoning workspace. This is where the model explores possibilities before committing to z_H.

Both states are initialized as learned embeddings at the start. z_H gets updated rarely (3 times), while z_L gets updated frequently (18 times) during the 21 recursive passes.

**5. Recursive processing**: This is the heart of TRM. The same 2-layer transformer processes the input 21 times in a carefully orchestrated pattern. Each pass refines z_L or z_H based on the current state of both latent variables and the input. We'll break down this structure in the next sections.

**6. Output prediction**: After 21 passes, z_H contains the refined solution. A final linear layer (called the "reverse embedding") projects each position in z_H back to token probabilities - 11 classes per cell. The model predicts the most likely token for each position.

**The key insight**: TRM trades space for time. Instead of having billions of parameters (space), it uses the same small network 21 times (time). It's like the difference between having 21 different consultants give you advice once, versus having one really good consultant iterate on their advice 21 times. Sometimes the second approach works better.

---

## The Big Idea: Weight Reuse

Okay, this is where things get interesting. This one design choice explains pretty much everything about why TRM works so well.

![type:video](assets/videos/Scene4_RecursionComparison.mp4)

The visualization makes this crystal clear:

**Traditional approach** (left side): Stack 32 unique transformer blocks, each with its own parameters. This is how HRM (TRM's predecessor) worked - different weights for each "layer" of reasoning. Total: 109M parameters (27M for HRM after optimizations), 55% accuracy on Sudoku-Extreme.

**TRM approach** (right side): Use 2 transformer blocks repeatedly, 21 times each. Same weights, multiple passes. Total: 7M parameters, 87.4% accuracy.

The numbers are kind of ridiculous: 15.6× fewer parameters, 1.6× better accuracy. But it's not just about being smaller and cheaper - it's about how the model actually learns.

**Why does weight reuse work so well?**

When you force a network to reuse the same weights across multiple passes, you're basically saying "you can't cheat by memorizing specific patterns for each step." The network can't learn "do X on the first pass, do Y on the second pass, do Z on the third." Nope. It has to learn ONE general operation that works for all passes.

It's like learning to edit your own writing. If you had a different editor for your first draft, second draft, and third draft, each one could develop their own weird specialized tricks. But if the SAME editor has to handle all three drafts? They better learn general principles of good editing that work iteratively. That's what TRM is doing.

This has a fancy name: regularization through compression. By cramming knowledge into fewer parameters, you prevent overfitting. The model literally cannot memorize "if I see this exact pattern, output this." There's not enough room. It HAS to learn the actual underlying structure.

And here's where it connects to scaling laws. The Chinchilla paper showed there's an optimal model size for any compute budget. But TRM adds a twist - when you only have 1,000 training examples (even with augmentations), the optimal model is way smaller than you'd think. Because smaller models just... can't overfit as easily. They're forced to generalize.

![type:video](assets/videos/Scene4b_ArchitectureComparison.mp4)

This alternative view reinforces the concept: traditional stacked layers on the left, recursive loop on the right. Notice how the recursive approach creates effective depth (21 passes × 2 layers = 42 effective layers) without the parameter cost. You get the benefits of a deep network - gradual refinement, hierarchical features, iterative improvement - without the memory requirements or overfitting risks.

---

## The Heart of TRM: Two Thinking Spaces

This is the part that really made me go "oh, that's smart." TRM maintains two separate "streams" of thought that update at different rates. And honestly? This mirrors how we actually think through complex problems.

When you're solving something hard, you don't just refine your answer. You also build up internal reasoning - scratch work, if you will - that supports your answer. TRM does the same thing.

![type:video](assets/videos/Scene5_TwoLatentStates.mp4)

Watch the visualization carefully - it breaks down into three phases:

**Phase 1** introduces the two states:
- **z_H** (shown in red): Your hypothesis or current answer. This is the model's best guess at the solution - the actual Sudoku grid values it's predicting.
- **z_L** (shown in blue): Your reasoning or working memory. This is scratch space where the model explores constraints, checks consistency, and works through logical implications.

**Phase 2** shows an L-cycle: the model updates z_L six times while z_H stays frozen. This is where the magic happens - the model is thinking through possibilities WITHOUT committing to an answer. It's doing that "if this cell is 5, then that row can't have another 5, which means..." type reasoning. Pure exploration.

**Phase 3** shows an H-cycle: the reasoning (z_L) informs an update to the answer (z_H). After all that exploration, the model makes one update to its draft solution based on what it figured out.

**The key numbers**:
- z_H gets updated only 3 times in the full 21 passes (via H-cycles)
- z_L gets updated 18 times (via L-cycles)
- Each L-cycle = 6 passes of updating z_L while z_H stays fixed
- Then 1 H-update where z_L informs z_H
- Total structure: 3 H-cycles × (6 L-cycles + 1 H-update) = 3 × 7 = 21 passes

**Why two states? Why not one, or three, or ten?**

At first I thought this seemed kind of arbitrary. But it makes sense when you think about what each state actually does.

**z_H is your commitment**. It's the model's current answer draft. Stop the model at any point and ask "what's your solution?" - that's z_H. It only updates 3 times because you don't want to thrash around with constant commitment changes. That'd be unstable.

**z_L is your scratch paper**. This is where the model tests ideas, checks constraints, explores possibilities - without worrying about maintaining a complete solution. It updates 18 times because exploration should be cheap and frequent. Try lots of stuff, commit to little.

And here's the thing - if you only had one state, every exploration would overwrite your answer. With two states, you can freely explore in z_L while keeping your stable solution in z_H. It's actually kind of elegant.

This design solves a fundamental problem in iterative reasoning: how do you explore possibilities without losing your current best answer? If you only had one state, each exploration would overwrite your previous solution. With two states, you can explore freely in z_L while keeping your stable solution in z_H.

**Connection to HRM's hierarchical interpretation:**

The predecessor paper (HRM) had all these biological arguments about the brain operating at different temporal frequencies. Honestly? TRM just... simplifies all that. It's not about hierarchy or biology. It's about having one place for stable answers and another place for messy exploration.

The paper actually tested using more states. With 7 states (one per recursion level), accuracy drops to 77.6%. With just one state, it drops to 71.9%. Two states hits 87.4%. Sweet spot confirmed.

**What's actually in these states?**

Here's something cool - if you decode z_H back through the reverse embedding, you can literally see the current Sudoku grid. It's interpretable. Real numbers in real cells.

z_L though? It's not directly interpretable. It's a latent representation - weird patterns of activation that somehow encode constraints and logical relationships. The model figures out what to put there during training. But the paper shows (Figure 6) that z_L definitely contains different information than z_H. It's not just a copy - it's genuinely doing its own thing.

This separation is what makes TRM more than just a recursive model. It's a model with an explicit internal reasoning process.

---

## The 21-Pass Structure

Let's zoom out and see how those L-cycles and H-cycles combine into the complete architecture.

![type:video](assets/videos/Scene6_TwentyOnePasses.mp4)

The structure is elegant in its simplicity:

**Big picture**: 3 H-cycles, each containing 7 passes, total 21 passes through the network. The same 2-layer transformer is used for all 21 passes.

**Inside each H-cycle**:
- 6 L-cycle passes: Network processes [input, z_H, z_L] → updates only z_L
- 1 H-update pass: Network processes [input, z_H, z_L] → updates only z_H
- Total: 7 passes per H-cycle

**Training trick - gradient flow**: Only H-Cycle 2 (the final one, passes 15-21) receives gradients during training. H-Cycles 0 and 1 (passes 1-14) run forward-only as "warmup."

**Why this design solves a critical problem:**

The naive approach would be to backpropagate through all 21 passes. But this creates a massive memory problem. Each pass requires storing activations for backpropagation. With 21 passes, you'd need 21× the memory of a single forward pass. For comparison, training GPT-3 sized models already pushes memory limits.

TRM's solution is "deep supervision" - supervise the output after deep processing, but only backpropagate through the final cycle. The first two H-cycles (14 passes) run in "warmup" mode. They update z_L and z_H, but those updates don't receive gradients. Think of them as preprocessing the problem before the model starts learning.

Then in the final H-cycle (7 passes), gradients flow backwards through all operations. The model learns how to take a preprocessed state and refine it to the solution.

This gives you:
- **Memory savings**: 3× reduction (7 passes with gradients vs 21)
- **Effective depth**: The model still gets 21 passes of iterative refinement
- **Better exploration**: Early passes can explore without gradient-driven constraints

The effective depth here is 42 layers (2 transformer layers × 21 passes). Compare this to typical transformers: GPT-2 has 12-48 layers, GPT-3 has 96 layers. TRM achieves similar effective depth with 6.8M parameters instead of billions.

---

## Inside the Transformer Blocks

We keep saying "2 transformer blocks," but what's actually inside them? And why only 2 layers instead of the typical 4, 12, or more?

![type:video](assets/videos/Scene7_TransformerDetails.mp4)

The visualization breaks down the architecture of each block:

**Attention mechanism** (for TRM-Att variant):
- 8 attention heads, each with 64 dimensions (8 × 64 = 512 total)
- Query-Key-Value projections: 786K parameters
- RoPE (Rotary Position Embeddings) handles positional information
- Attention is non-causal (each token can attend to all tokens, not just previous ones)

Why RoPE instead of learned positional embeddings? RoPE encodes position as rotation in the embedding space, which generalizes better to different sequence lengths and requires no additional parameters.

**MLP (feed-forward network)**:
- SwiGLU activation function (a gated variant more powerful than ReLU)
- Gate-up projection: 1.57M parameters (expands 512 dim → 2048 dim)
- Down projection: 0.79M parameters (compresses 2048 dim → 512 dim)
- RMSNorm for layer normalization (simpler, slightly faster than LayerNorm)

**Total per block**: About 3.4M parameters (1M for attention + 2.4M for MLP)

**Two blocks stacked**: 6.8M parameters total

These blocks get reused 21 times, creating 42 effective layers from just 6.8M physical parameters.

**Why 2 layers is optimal:**

The ablation study shows something kind of mind-blowing: 4-layer blocks get 79.5% accuracy. 2-layer blocks get 87.4%. Wait, what? Smaller is better?

Yeah. With only 1,000 training examples (even with augmentations), 4-layer blocks are just too big. They start memorizing specific puzzles instead of learning how Sudoku actually works. The 2-layer blocks don't have enough capacity to memorize, so they're forced to actually learn the rules.

This connects to a broader principle: when data is limited, smaller models trained longer beat larger models trained less. And TRM trains for about 1 million optimizer steps. That's a LOT for a 7M parameter model. But it works.

**TRM-MLP variant:**

Interestingly, the paper also presents a variant called TRM-MLP that replaces self-attention with an MLP that operates on the sequence dimension. This works well for Sudoku (87.4% vs 74.7% for attention-based), but poorly on ARC-AGI and mazes. The lesson: for highly structured, fixed-size grids, position-wise MLPs can be more efficient than attention. But attention is needed when the task structure varies.

---

## Deep Supervision: Memory-Efficient Training

Here's a clever training strategy that makes TRM practical. This section connects deeply to the 21-pass structure, so let's see how gradient flow works in detail.

![type:video](assets/videos/Scene8_DeepSupervision.mp4)

The visualization shows three H-cycles with `detach()` operations between them - these are the points where gradient flow is broken.

**H-Cycles 0 & 1** (grayed out): These run forward-only. The model processes the input, updates z_L and z_H, but doesn't compute or store gradients. In PyTorch terms, these operations happen inside a `with torch.no_grad():` block. No memory cost for storing activations, no computational cost for backpropagation.

**H-Cycle 2** (highlighted): Full backpropagation happens here. Gradients flow backward through all 7 passes (6 L-cycles + 1 H-update), updating the transformer weights based on the final loss.

**The benefits**:
- **Memory**: 3× reduction - store activations for 7 passes instead of 21
- **Speed**: Backprop is the expensive part of training. By only backpropping through 7 passes, training is roughly 2× faster than full backprop through 21 passes.
- **Quality**: Early cycles explore without gradient constraints, then the final cycle learns from that exploration

**This is called "deep supervision"** because you're supervising the output after deep recursive processing (21 passes), but only backpropagating through the final cycle.

**Connection to HRM and the IFT controversy:**

HRM (TRM's predecessor) used something called the "Implicit Function Theorem with 1-step gradient approximation." The idea was that if the recursive process converges to a fixed point, you can approximate gradients through all passes by only backpropping through the last step.

But there's a problem: HRM never actually verified that a fixed point is reached. The paper shows (Figure 3 in Wang et al. 2025) that residuals remain non-zero even after many passes. Using IFT without convergence is theoretically questionable.

TRM's approach is simpler and more honest: don't try to approximate gradients through early passes. Just run them forward-only, and fully backprop through the final passes where you actually want to learn. This is more memory-efficient than HRM's approach (which required storing some activations from early passes for the 1-step approximation), and empirically works better (87.4% vs 55% on Sudoku-Extreme).

**An interesting connection to deep equilibrium models:**

This approach is reminiscent of Deep Equilibrium Models (DEQ), which solve for fixed points and backprop through them implicitly. But TRM doesn't try to reach equilibrium - it explicitly performs iterative refinement. The warmup cycles effectively give the model a "head start" before training kicks in.

---

## Adaptive Computation: Matching Effort to Difficulty

During training, TRM learns to adapt how much computation it uses based on problem difficulty. This is purely a training optimization - it doesn't affect inference.

![type:video](assets/videos/Scene9_AdaptiveComputation.mp4)

The visualization contrasts:
- **Easy problems**: 2-4 supervision steps might be enough to reach correct solution
- **Hard problems**: Need 12-16 supervision steps to solve

The Q-halt mechanism tracks confidence: starting at -4.2 (uncertain, keep going) and increasing to +0.5 (confident, can stop).

**What is ACT (Adaptive Computation Time)?**

Without ACT, training would run all N_sup=16 supervision steps for every example. But many examples are easy - after 2-3 steps, the model already has the right answer. Running 13 more steps wastes computation.

ACT learns to predict when to stop. The model has an additional head (separate from the main output) that predicts a "halting probability" after each supervision step. If the halt probability exceeds a threshold, training moves to the next example.

**TRM's simpler approach vs HRM:**

HRM used Q-learning for ACT, which required two forward passes per training step:
1. Current pass: predict answer and halt value
2. Extra pass: predict next-step halt value (for the Q-learning "continue" loss)

TRM simplifies this to binary classification: after each supervision step, predict whether the current answer is correct. Train with binary cross-entropy loss against the ground truth. Only one forward pass needed.

In practice on Sudoku-Extreme, this reduces average supervision steps from 16 to under 2 during training. The model spends more iterations on hard examples (which is good - they need more training), while breezing through easy examples (which is efficient - they don't need more training).

**Important note**: ACT is only used during training to be more efficient. At inference time, TRM always runs the full 16 supervision steps to ensure consistent, maximum performance. There's no early stopping at test time.

---

## Watching TRM Solve a Puzzle

Let's see everything come together in one end-to-end demonstration. This visualization connects all the concepts we've covered.

![type:video](assets/videos/Scene10_SolvingProcess.mp4)

The visualization shows a complete forward pass through TRM:

**Initial state**: A 70% filled Sudoku grid (about 25 cells given, 56 cells to be solved) with randomly initialized z_H and z_L tensors. The input grid is tokenized and embedded. z_H is initialized with a learned embedding that roughly represents "no answer yet." z_L is initialized similarly.

**Warmup phase - H-Cycles 0 and 1**: Watch the blue (z_L) and red (z_H) tensors pulse as they're updated. During these 14 passes:
- z_L updates 12 times (6 per H-cycle), building up reasoning about constraints and possibilities
- z_H updates 2 times (once per H-cycle), gradually forming a draft solution
- No gradients flow - this is pure forward inference

After the warmup, z_H already contains a rough solution (though possibly with errors), and z_L contains accumulated reasoning.

**Final phase - H-Cycle 2**: The supervised cycle, where learning happens. Another 7 passes refine both states:
- z_L gets 6 more updates, refining reasoning
- z_H gets 1 final update, producing the final answer
- Gradients flow backward through all 7 operations

**Final state**: The tensors turn green, indicating successful completion. z_H now contains the complete, correct solution. If you were to decode it through the reverse embedding, you'd get all 81 cells filled correctly.

**This is the 87.4% accuracy in action** - taking a partially filled puzzle and reasoning through to the complete solution through 21 iterative passes.

**What's happening under the hood:**

During the L-cycles, the model is effectively checking constraints: "If cell (3,4) is 7, then cell (3,7) can't be 7, which means..." This constraint propagation happens in the latent space of z_L.

During the H-updates, the model commits to answers: "Based on all the reasoning in z_L, cell (3,4) should be 7, and cell (3,7) should be 2."

The iterative structure lets the model fix mistakes. If an early H-update sets cell (3,4) to the wrong value, subsequent L-cycles can detect the inconsistency, and the next H-update can correct it.

This iterative refinement - think, answer, think more, refine answer - is fundamentally different from how LLMs work (generate answer autoregressively, no refinement). That's why TRM succeeds where LLMs fail on Sudoku.

---

## Results: How Well Does It Work?

Let's look at the numbers across different benchmarks, with proper context for each result:

| Benchmark | TRM | HRM | Others |
|-----------|-----|-----|--------|
| Sudoku-Extreme | **87.4%** (TRM-MLP) | 55% | 0% (GPT-4, Claude, DeepSeek R1) |
| ARC-AGI-1 | **44.6%** (TRM-Att) | 40.3% | 21% (direct prediction) |
| ARC-AGI-2 | **7.8%** (TRM-Att) | 5.0% | 4.9% (Gemini 2.5 Pro) |
| Maze-Hard | **85.3%** (TRM-Att) | 74.5% | - |

**Understanding these benchmarks:**

**Sudoku-Extreme**: A dataset of extremely difficult Sudoku puzzles where most given cells are at the minimum (17 givens for 9×9 Sudoku). Only 1,000 training examples are used, but tested on 423,000 examples. The fact that TRM trained on 1K examples generalizes to 423K test cases shows remarkable generalization.

**ARC-AGI-1 and ARC-AGI-2**: The Abstraction and Reasoning Corpus is a benchmark designed to test abstract reasoning - the kind humans excel at but AI struggles with. Each task shows 2-3 input-output examples of a transformation rule (like "rotate 90 degrees then change red to blue"), and the model must apply that rule to new inputs. ARC-AGI-2 (released in 2025) is significantly harder than ARC-AGI-1.

For context, human performance on ARC-AGI-1 is around 85%. TRM achieves 44.6%, which is impressive for a 7M parameter model. The best LLM results (with heavy test-time compute) reach 37-67% depending on the model and compute budget.

**Maze-Hard**: 30×30 mazes where the shortest path exceeds 110 steps. Both training and test sets have only 1,000 mazes each. This tests whether the model can learn spatial reasoning and pathfinding from limited data.

**Why TRM-MLP works better for Sudoku:**

TRM-MLP replaces self-attention with an MLP operating on the sequence dimension. For Sudoku, the grid structure is fixed (always 9×9), and spatial relationships are predetermined (rows, columns, boxes). An MLP can hardcode these spatial relationships through learned weights. Attention is more flexible but less efficient for this specific structure.

For ARC-AGI and mazes, grid sizes vary (up to 30×30) and spatial relationships are task-dependent. Here, attention's flexibility is needed, which is why TRM-Att performs better.

**Comparison to HRM:**

The most meaningful comparison is to HRM, since both target the same problems with similar approaches. TRM consistently outperforms:
- Sudoku: 87.4% vs 55% (59% relative improvement)
- ARC-AGI-1: 44.6% vs 40.3% (11% improvement)
- ARC-AGI-2: 7.8% vs 5.0% (56% improvement)
- Maze: 85.3% vs 74.5% (14% improvement)

All while using 3.9× fewer parameters (7M vs 27M).

**The training setup:**

These results come from training on limited data with heavy augmentation:
- **Sudoku-Extreme**: 1,000 base examples × 1,000 augmentations (rotations, reflections, number shuffling) = ~1M training examples
- **ARC-AGI**: 800 training tasks + 400 evaluation tasks (used for training) + 160 ConceptARC tasks = 1,360 tasks × 3 examples per task × 1,000 augmentations = ~4M training examples
- **Maze**: 1,000 mazes × 8 dihedral transformations = 8,000 training examples

Training runs for about 1 million optimizer steps (roughly 2 days on 4× H100 GPUs). The extensive training on small data with heavy augmentation is what enables generalization.

---

## Why TRM Works: Key Insights

Several architectural choices combine to make TRM effective:

**Weight reuse** forces compression. By using the same 6.8M parameters 21 times, the network must learn general reasoning operations. There's no room for memorizing specific patterns.

**Dual latent states** separate reasoning from refinement. z_L provides 18 updates of scratch space for exploration. z_H provides 3 stable updates toward the final answer. This mirrors how humans think through problems.

**Deep supervision** saves memory while maintaining quality. Training only the final H-cycle reduces memory 3×, while early cycles still contribute via forward-only warmup.

**2 layers > 4 layers** for small datasets. Experiments show that with limited data (like 1000 Sudoku examples), 2-layer blocks outperform 4-layer blocks. Fewer parameters prevent overfitting.

### Compared to HRM

TRM simplifies and improves upon HRM:

| Feature | HRM | TRM | Improvement |
|---------|-----|-----|-------------|
| Parameters | 27M | 7M | 3.9× reduction |
| Networks | 2 separate (f_L, f_H) | 1 unified | Simpler |
| Layers per block | 4 | 2 | Less overfitting |
| Gradient flow | 1-step approximation | Full backprop | Better training |
| ACT mechanism | Q-learning (2 passes) | Binary classification (1 pass) | More efficient |
| Sudoku accuracy | 55% | 87.4% | 59% improvement |

The takeaway: simplification improved performance. Fewer networks, fewer layers, simpler training - all led to better results.

---

## Want to Try It Yourself?

The complete implementation is available in the [TinyRecursiveModels GitHub repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) by Samsung SAIL Montreal.

**Key files to explore**:
- [`models/recursive_reasoning/trm.py`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/trm.py): Core TRM architecture implementing the dual latent states (z_H and z_L), L-cycles, H-cycles, and deep supervision
- [`models/recursive_reasoning/hrm.py`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/recursive_reasoning/hrm.py): HRM implementation for comparison
- [`pretrain.py`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/pretrain.py): Training script with deep supervision, ACT mechanism, and EMA
- [`puzzle_dataset.py`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/puzzle_dataset.py): Dataset handling, tokenization, and augmentation strategies
- [`models/layers.py`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/layers.py): Transformer building blocks (attention, SwiGLU, RoPE)

**Paper and additional resources**:
- **TRM Paper**: ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871) (arXiv 2510.04871) - full technical details, ablation studies, and theoretical analysis
- **HRM Paper**: ["Hierarchical Reasoning Model"](https://arxiv.org/abs/2506.21734) (arXiv 2506.21734) - TRM's predecessor
- **Nature Article**: ["'Tiny' AI model beats massive LLMs at logic test"](https://www.nature.com/articles/d41586-025-03379-9) - external coverage validating the results
- **ARC-AGI Benchmark**: [arcprize.org](https://arcprize.org/) - details on the benchmark and leaderboard
- **Educational Video**: [YouTube walkthrough by ARC-AGI researcher](https://www.youtube.com/watch?v=yJQQB6MIUd0) (referenced in this post) - explains TRM concepts from a competition perspective

**Training requirements**:

If you want to replicate the paper's results:
- **Compute**: 4× H100 GPUs for about 2 days (or 8× A100s for ~60 hours)
- **Memory**: Each GPU needs ~40GB VRAM for batch size 32
- **Dataset**: Training data is generated programmatically (code's in the repo)

For smaller-scale experiments, you can train on Sudoku with a single A100 or 4090 in about 6-8 hours if you reduce the batch size. Still totally doable.

---

## What This Actually Means

Look, I think TRM is impressive. But let's be honest about what it is and isn't. It shows that for certain types of problems - structured reasoning with limited data - clever architecture beats raw parameter scaling. But there are caveats.

**Where TRM excels:**

The dual latent states (z_H for answers, z_L for reasoning) enable progressive problem-solving similar to human cognition. We work through possibilities before committing to answers. This design works exceptionally well for problems where:
- The solution space is well-defined (complete Sudoku grids, valid maze paths)
- Iterative refinement is natural (fix mistakes, check constraints)
- Solutions can be verified locally (cell-by-cell checking)
- Training data is limited but augmentable

**The "less is more" insight:**

Weight reuse creates deep effective networks (42 layers) without the parameter cost (7M). Forcing the same small network to handle all 21 passes means it learns general reasoning operations rather than task-specific patterns. This is regularization through architectural constraint.

But this only works when the data regime is small. With millions of diverse examples, larger models would likely perform better. TRM's strength is specifically in the low-data, structured-reasoning regime where most models overfit.

**Limitations and stuff that's still unclear:**

TRM is NOT a general-purpose model. It's not going to replace LLMs. It's designed for specific types of problems. Here's what you need to know:

- **Grid-based only**: The whole tokenization assumes 2D grids. You can't just throw natural language at this. Different problem = different architecture needed.
- **One solution per problem**: TRM predicts a single deterministic solution. Creative tasks? Open-ended generation? Not happening without major changes.
- **Compute trade-off**: Yeah, it has fewer parameters. But 21 forward passes means you're doing 21× more compute at inference than a single-pass model. Training takes 2 days on 4× H100s. That's not nothing.
- **Task-specific**: A trained TRM is specialized. Unlike GPT-4 which handles everything from code to poetry, TRM does one thing well.

**Open questions that remain:**
- How does this generalize beyond grid-based puzzles? Could similar principles apply to code generation, mathematical proofs, or scientific reasoning?
- What's the optimal recursion depth for different problems? The paper uses 21 passes, but is this optimal for all tasks?
- Could these ideas integrate with larger language models? A hybrid system with LLM-based reasoning (z_L) and structured answer generation (z_H) might combine strengths of both approaches.
- Does the dual-state idea apply to other domains? Medical diagnosis might benefit from separate "hypothesis" and "evidence" states that update at different rates.

**The broader significance:**

TRM offers evidence that we don't always need bigger models. When data is limited and the problem is structured, smaller models with thoughtful architectures can outperform giants. This matters for:
- **Research labs with limited compute**: You don't need hundreds of GPUs to do important AI research
- **Edge deployment**: 7M parameters can run on phones and embedded devices
- **Environmental impact**: Smaller models use less energy for training and inference
- **Scientific understanding**: Simpler models are easier to analyze and understand

The lesson here isn't "small models are always better." They're not. The lesson is "match your architecture to your problem." For grid-based reasoning with 1K examples? TRM is near-optimal. For general text understanding with billions of examples? LLMs win.

TRM shows us that the future of AI isn't just "make it bigger." It's about understanding what you're trying to solve and designing something that actually fits that problem. That's true whether you've got hundreds of GPUs or just one.

And honestly? I think that's a more interesting direction than just throwing more compute at everything.

---

**Further Reading & Sources:**

**Primary Sources:**
- [TRM Paper: "Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871) - Jolicoeur-Martineau et al., arXiv 2510.04871
- [TinyRecursiveModels GitHub Repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) - Official implementation by Samsung SAIL Montreal
- [HRM Paper: "Hierarchical Reasoning Model"](https://arxiv.org/abs/2506.21734) - Wang et al., arXiv 2506.21734 (TRM's predecessor)

**Benchmarks & Competitions:**
- [ARC-AGI Benchmark](https://arcprize.org/) - Competition homepage and leaderboard
- [ARC-AGI Prize Analysis](https://arcprize.org/blog/hrm-analysis) - Technical analysis of HRM performance

**External Coverage:**
- [Nature Article on TRM](https://www.nature.com/articles/d41586-025-03379-9) - "'Tiny' AI model beats massive LLMs at logic test"
- [Educational YouTube Walkthrough](https://www.youtube.com/watch?v=yJQQB6MIUd0) - Deep dive by ARC-AGI researcher

**Related Work:**
- [Chinchilla Scaling Laws Paper](https://arxiv.org/abs/2203.15556) - Optimal model sizing principles
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) - Related work on fixed-point recursion
- [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) - Alternative approach to improving reasoning

---

*This is part of AI Engineering Academy's AI Breakdown series, where I take important research papers and break them down with visualizations and (hopefully) clear explanations. I created all the visualizations for this post using Remotion - check them out throughout the article.*

*If you found this useful, we cover more topics like this at [AI Engineering Academy](https://aiengineering.academy). We're building a place for practical AI engineering knowledge, not just theory.*
