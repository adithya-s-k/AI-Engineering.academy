# Tiny Recursive Models (TRM): When Small Models Think Better

Something fascinating is happening in AI research. A 7 million parameter model is outperforming models 100,000 times its size on certain reasoning tasks. Not by a little - by a lot. On Sudoku puzzles, it scores 87.4% while models like GPT-4 and Claude score 0%.

This isn't about making models bigger. It's about making them think differently.

Let me walk you through how TRM works, using visualizations to make the concepts clear. By the end, you'll understand why recursive refinement might be more powerful than just throwing more parameters at a problem.

---

## What is TRM?

TRM (Tiny Recursive Models) is a research breakthrough for structured reasoning tasks - things like Sudoku puzzles, mazes, and abstract reasoning problems (ARC-AGI). Instead of processing a problem once with a massive network, TRM processes it many times with a tiny network.

Think about how you solve a Sudoku puzzle. You don't look at it once and write down the complete answer. You fill in some numbers, check your work, reconsider, make corrections, and iterate. TRM does exactly this.

![type:video](assets/videos/Scene1_Title.mp4)

The results speak for themselves:
- 87.4% on Sudoku-Extreme (vs 55% for its predecessor HRM, 0% for standard LLMs)
- 44.6% on ARC-AGI-1 (a hard abstract reasoning benchmark)
- Only 7M parameters - you could run this on a laptop

---

## The Problem Domain: Why Sudoku?

Let's start with a concrete example. Sudoku is perfect for understanding TRM because it needs both local reasoning (checking if a number fits in a cell) and global reasoning (ensuring the whole grid is consistent).

![type:video](assets/videos/Scene2_SudokuSetup.mp4)

The visualization shows two things:

**First**, you see an input puzzle on the left (with dashes for empty cells) and the solved version on the right. This is what TRM learns to do: take a partial grid and fill in the gaps while respecting all the rules.

**Second**, you see how the 9×9 grid gets transformed into a sequence of tokens. Machine learning models don't understand visual grids directly - they need everything as numbers in a sequence. So each cell becomes a token, and the grid becomes a string of 81 tokens.

The key insight: this same approach works for any grid-based reasoning task. Mazes, abstract visual puzzles (ARC-AGI), even things like game boards. If you can represent it as a grid, TRM can learn to reason about it.

---

## How Data Flows Through TRM

Before we get to the clever parts, let's trace how information moves through the model from start to finish.

![type:video](assets/videos/Scene3_DataFlow.mp4)

Follow the pipeline from left to right:

1. **Input tokens** (81 for a 9×9 Sudoku) get embedded into 512-dimensional vectors
2. A **puzzle ID** gets added - this lets the model know which examples belong to the same task
3. The combined input becomes a sequence of length 97 (with padding and special tokens)
4. Here's where it gets interesting: the model maintains **two latent states** - z_H and z_L (more on these soon)
5. The same small network processes these states **21 times** recursively
6. Finally, outputs become predictions for each cell (11 possible classes: padding, end-of-sequence, and digits 0-9)

This pipeline might look complex, but the beauty is in step 5: instead of having 21 different networks, we use the same tiny network 21 times. That's the core innovation.

---

## The Big Idea: Weight Reuse

Here's where TRM fundamentally differs from traditional approaches.

![type:video](assets/videos/Scene4_RecursionComparison.mp4)

The visualization makes this crystal clear:

**Traditional approach** (left side): Stack 32 unique transformer blocks, each with its own parameters. Total: 109M parameters, 55% accuracy on Sudoku.

**TRM approach** (right side): Use 2 transformer blocks repeatedly, 21 times each. Total: 7M parameters, 87.4% accuracy.

The math is striking: 15.6× fewer parameters, 1.6× better accuracy.

Why does this work? When you force a network to reuse the same weights over and over, it has to learn general reasoning operations rather than task-specific tricks. It's like the difference between memorizing specific solutions versus learning the underlying principles.

![type:video](assets/videos/Scene4b_ArchitectureComparison.mp4)

This alternative view shows the same concept differently: traditional stacked layers on the left, recursive loop on the right. The recursive approach creates effective depth without the parameter cost.

---

## The Heart of TRM: Two Thinking Spaces

This is the most important concept to understand. TRM maintains two separate "streams" of thought that update at different rates.

![type:video](assets/videos/Scene5_TwoLatentStates.mp4)

Watch the visualization carefully - it breaks down into three phases:

**Phase 1** introduces the two states:
- **z_H** (shown in red): Your hypothesis or current answer. This is the model's best guess at the solution.
- **z_L** (shown in blue): Your reasoning or working memory. This is scratch space for thinking through the problem.

**Phase 2** shows an L-cycle: the model updates z_L six times while keeping z_H frozen. Think of this as working through possibilities in your head before committing to an answer.

**Phase 3** shows an H-cycle: the reasoning (z_L) informs an update to the answer (z_H). Your scratch work influences your final answer.

The key numbers:
- z_H gets updated only 3 times in the full 21 passes (via H-cycles)
- z_L gets updated 18 times (via L-cycles)
- Each L-cycle = 6 passes of updating z_L while z_H stays fixed
- Then 1 H-update where z_L informs z_H

Why two states? Because reasoning and answer refinement are different processes. By keeping them separate, the model can explore many reasoning paths (via z_L) before committing to answer updates (via z_H).

---

## The 21-Pass Structure

Let's zoom out and see how those L-cycles and H-cycles combine.

![type:video](assets/videos/Scene6_TwentyOnePasses.mp4)

The structure is elegant:

**Big picture**: 3 H-cycles, each containing 7 passes, total 21 passes through the network.

**Inside each H-cycle**: 6 L-cycle passes (updating z_L) + 1 H-update (updating z_H) = 7 passes per H-cycle.

**Training trick**: Only H-Cycle 2 (the final one) receives gradients during training. The first two H-cycles run forward-only as "warmup."

Why this design? The early cycles explore different reasoning paths without being constrained by gradients. Only the final cycle gets trained, saving memory and letting early exploration happen freely.

---

## Inside the Transformer Blocks

We keep saying "2 transformer blocks," but what's actually inside them?

![type:video](assets/videos/Scene7_TransformerDetails.mp4)

The visualization breaks it down:

**Attention mechanism**:
- 8 attention heads, each with 64 dimensions
- Query-Key-Value projections: 786K parameters
- RoPE (Rotary Position Embeddings) handles positional information

**MLP (feed-forward network)**:
- SwiGLU activation function
- Gate-up projection: 1.57M parameters
- Down projection: 0.79M parameters

**Total per block**: About 3.4M parameters (1M for attention + 2.4M for MLP)

**Two blocks total**: 6.8M parameters

These blocks get reused 21 times, creating 42 effective layers from just 6.8M physical parameters. It's this reuse that creates both the efficiency and the regularization benefits.

---

## Deep Supervision: Memory-Efficient Training

Here's a clever training strategy that makes TRM practical.

![type:video](assets/videos/Scene8_DeepSupervision.mp4)

The visualization shows three H-cycles with `detach()` operations between them - breaking the gradient flow.

H-Cycles 0 & 1 (grayed out): These run forward-only. No gradients, no memory cost for storing activations.

H-Cycle 2 (highlighted): Full backpropagation happens here. Gradients flow backward through these 7 passes.

The benefits:
- **Memory**: 3× reduction - store activations for 7 passes instead of 21
- **Quality**: Early cycles explore without gradient constraints, then the final cycle learns from that exploration

This is called "deep supervision" because you're supervising the output after deep recursive processing, but only backpropagating through the final cycle.

---

## Adaptive Computation: Matching Effort to Difficulty

During training, TRM learns to adapt how much computation it uses based on problem difficulty.

![type:video](assets/videos/Scene9_AdaptiveComputation.mp4)

The visualization contrasts:
- Easy problems: 2-4 steps might be enough
- Hard problems: Need 12-16 steps to solve

The Q-halt mechanism tracks confidence: starting at -4.2 (uncertain, keep going) and increasing to +0.5 (confident, can stop).

Important note: This is only used during training to be more efficient. At inference time, TRM always runs the full 16 refinement steps to ensure consistent performance.

---

## Watching TRM Solve a Puzzle

Let's see everything come together in one end-to-end demonstration.

![type:video](assets/videos/Scene10_SolvingProcess.mp4)

The visualization shows:

**Initial state**: A 70% filled Sudoku grid with initialized z_H and z_L tensors.

**Warmup phase**: H-Cycles 0 and 1 run, with the blue (z_L) and red (z_H) tensors pulsing to show the model's internal reasoning.

**Final phase**: The supervised cycle fills the remaining cells. The tensors turn green, indicating successful completion and a fully solved grid.

This is the 87.4% accuracy in action - taking a partially filled puzzle and reasoning through to the complete solution.

---

## Results: How Well Does It Work?

Let's look at the numbers across different benchmarks:

| Benchmark | TRM | HRM | Others |
|-----------|-----|-----|--------|
| Sudoku-Extreme | **87.4%** | 55% | 0% (standard LLMs) |
| ARC-AGI-1 | **44.6%** | 40.3% | 21% (direct prediction) |
| ARC-AGI-2 | **7.8%** | 5.0% | 4.9% (Gemini 2.5 Pro) |
| Maze-Hard | **85.3%** | 74.5% | - |

A few notes on these results:

"TRM-MLP" (used for Sudoku) uses only feed-forward layers without self-attention. "TRM-Att" (used for mazes and ARC-AGI) includes attention. The choice depends on the task structure.

The comparison to HRM (TRM's predecessor) is most meaningful - same problem domain, both designed for structured reasoning. TRM achieves better results with 3.9× fewer parameters.

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
- `model.py`: TRM architecture with dual latent states
- `train.py`: Training loop with deep supervision
- `data.py`: Grid tokenization and data augmentation

**Paper**: The full technical details, ablation studies, and theoretical analysis are in the [arXiv paper (2510.04871)](https://arxiv.org/abs/2510.04871).

---

## What This Means

TRM demonstrates something important: for structured reasoning tasks with limited data, architectural innovation beats parameter scaling.

The dual latent states (z_H for answers, z_L for reasoning) enable progressive problem-solving similar to human cognition. We work through possibilities before committing to answers.

Weight reuse creates deep effective networks (42 layers) without the parameter cost (7M). Forcing the same small network to handle all 21 passes means it learns general principles rather than specific patterns.

These techniques are particularly effective when you have limited training data. TRM trains on about 1,000 Sudoku examples and generalizes well. Larger models would overfit.

Open questions remain: How does this generalize beyond grid-based puzzles? What's the optimal recursion depth for different problems? Could these ideas integrate with larger language models for improved reasoning?

For now, TRM offers compelling evidence that we don't always need bigger models. Sometimes we need smarter architectures that think more like we do - iteratively, with separate spaces for reasoning and answers, building understanding through refinement.

---

**Further Reading:**
- [TRM Paper: "Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2510.04871) (arXiv 2510.04871)
- [TinyRecursiveModels GitHub Repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [HRM Paper: "Hierarchical Reasoning Model"](https://arxiv.org/abs/2506.21734) (TRM's predecessor)
- [ARC-AGI Benchmark](https://arcprize.org/)
