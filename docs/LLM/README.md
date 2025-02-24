# Large Language Models (LLMs)

Welcome to the Large Language Models section of the AI Engineering Academy. This module provides a comprehensive understanding of LLMs and their practical applications in AI engineering.

## Repository Structure

| Category                      | Topic                                              | Resource                                                                           |
| ----------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Introduction**              | Overview                                           | [Introduction to LLMs](README.md)                                                  |
| **Theory Behind Fine-tuning** | Pre-Training                                       | [Pre-Training](TheoryBehindFinetuning/PreTrain.md)                                 |
|                               | Supervised Fine-Tuning (SFT)                       | [SFT Theory](TheoryBehindFinetuning/SFT.md)                                        |
|                               | Proximal Policy Optimization (PPO)                 | [PPO Theory](TheoryBehindFinetuning/PPO.md)                                        |
|                               | Direct Preference Optimization (DPO)               | [DPO Theory](TheoryBehindFinetuning/DPO.md)                                        |
|                               | Observation-Regularized Policy Optimization (ORPO) | [ORPO Theory](TheoryBehindFinetuning/ORPO.md)                                      |
|                               | Gated Regularized Policy Optimization (GRPO)       | [GRPO Theory](TheoryBehindFinetuning/GRPO.md)                                      |
| **Hands-On SFT**              | Overview                                           | [SFT Implementation Guide](HandsOnWithFinetuning/SFT/README.md)                    |
|                               | Implementation                                     | [SFT Notebook](HandsOnWithFinetuning/SFT/SFT_finetuning_notebook.ipynb)            |
| **Hands-On GRPO**             | Guide                                              | [Hacker Guide to GRPO](HandsOnWithFinetuning/GRPO/hacker_guide_to_GRPO.md)         |
|                               | Implementation                                     | [Qwen 0.5B GRPO](HandsOnWithFinetuning/GRPO/Qwen_0_5b__GRPO.ipynb)                 |
| **Gemma**                     | Overview                                           | [Gemma Guide](Gemma/README.md)                                                     |
|                               | Implementation                                     | [Gemma Fine-tuning](Gemma/Gemma_finetuning_notebook.ipynb)                         |
| **Llama2**                    | Overview                                           | [Llama2 Guide](LLama2/README.md)                                                   |
|                               | Implementation                                     | [Llama2 Fine-tuning](LLama2/Llama2_finetuning_notebook.ipynb)                      |
|                               | Advanced                                           | [QLora Fine-tuning](LLama2/Llama_2_Fine_Tuning_using_QLora.ipynb)                  |
| **Llama3**                    | Implementation                                     | [Llama3 Fine-tuning](Llama3_finetuning_notebook.ipynb)                             |
| **Mistral-7B**                | Overview                                           | [Mistral Guide](Mistral-7b/README.md)                                              |
|                               | Implementation                                     | [Mistral Fine-tuning](Mistral-7b/Mistral_finetuning_notebook.ipynb)                |
|                               | Evaluation                                         | [Evaluation Harness](Mistral-7b/LLM_evaluation_harness_for_Arc_Easy_and_SST.ipynb) |
|                               | DPO                                                | [DPO Fine-tuning](Mistral-7b/notebooks_DPO_fine_tuning.ipynb)                      |
|                               | SFT                                                | [SFT Trainer](Mistral-7b/notebooks_SFTTrainer%20TRL.ipynb)                         |
|                               | Inference                                          | [ChatML Inference](Mistral-7b/notebooks_chatml_inference.ipynb)                    |
| **Mixtral**                   | Implementation                                     | [Mixtral Fine-tuning](Mixtral/Mixtral_fine_tuning.ipynb)                           |
| **Visual Language Models**    | Florence2                                          | [Florence2 Fine-tuning](VLM/Florence2_finetuning_notebook.ipynb)                   |
|                               | PaliGemma                                          | [PaliGemma Fine-tuning](VLM/PaliGemma_finetuning_notebook.ipynb)                   |
| **Architecture**              | Parameter Analysis                                 | [Parameter Count](LLMArchitecture/ParameterCount/README.md)                        |

## Learning Roadmap

| Level            | Steps                        | Resources                                                                                                  |
| ---------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Beginner**     | 1. Introduction to LLMs      | [Introduction](README.md)                                                                                  |
|                  | 2. Understanding core theory | [Pre-Training](TheoryBehindFinetuning/PreTrain.md), [SFT Theory](TheoryBehindFinetuning/SFT.md)            |
|                  | 3. First implementation      | [SFT Guide](HandsOnWithFinetuning/SFT/README.md)                                                           |
|                  | 4. Practical application     | [Llama2 Fine-tuning](LLama2/Llama2_finetuning_notebook.ipynb)                                              |
| **Intermediate** | 1. Advanced techniques       | [DPO Theory](TheoryBehindFinetuning/DPO.md), [PPO Theory](TheoryBehindFinetuning/PPO.md)                   |
|                  | 2. Model implementation      | [Mistral Fine-tuning](Mistral-7b/Mistral_finetuning_notebook.ipynb)                                        |
|                  | 3. Architecture concepts     | [Parameter Count](LLMArchitecture/ParameterCount/README.md)                                                |
| **Advanced**     | 1. Cutting-edge methods      | [ORPO Theory](TheoryBehindFinetuning/ORPO.md), [GRPO Theory](TheoryBehindFinetuning/GRPO.md)               |
|                  | 2. Advanced implementation   | [GRPO Implementation](HandsOnWithFinetuning/GRPO/Qwen_0_5b__GRPO.ipynb)                                    |
|                  | 3. Multimodal models         | [Florence2](VLM/Florence2_finetuning_notebook.ipynb), [PaliGemma](VLM/PaliGemma_finetuning_notebook.ipynb) |

## Contributing

We welcome contributions to expand this repository. Please follow the standard pull request process and ensure your contributions align with the overall structure.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

_AI Engineering Academy - Advancing the frontier of language model understanding and implementation_

<!-- # 🤖 Large Language Models (LLMs)

Welcome to the Large Language Models section of the AI Engineering Academy! This module provides a comprehensive understanding of LLMs and their practical applications in AI engineering.

## 📚 Repository Structure

| Model/Directory                                                                                                                                                                                                                                | Description & Contents                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [**Axolotl**](./Axolotl/)                                                                                                                                                                                                                      | Framework for fine-tuning language models                          |
| [**Gemma**](./Gemma/)                                                                                                                                                                                                                          | Google's latest LLM implementation                                 |
| - `finetune-gemma.ipynb`<br>- `gemma-sft.py`<br>- `Gemma_finetuning_notebook.ipynb`                                                                                                                                                            | Fine-tuning notebooks and scripts                                  |
| [**LLama2**](./LLama2/)                                                                                                                                                                                                                        | Meta's open-source LLM                                             |
| - `generate_response_stream.py`<br>- `Llama2_finetuning_notebook.ipynb`<br>- `Llama_2_Fine_Tuning_using_QLora.ipynb`                                                                                                                           | Implementation and fine-tuning guides                              |
| [**Llama3**](./Llama3/)                                                                                                                                                                                                                        | Upcoming Meta LLM experiments                                      |
| - `Llama3_finetuning_notebook.ipynb`                                                                                                                                                                                                           | Initial fine-tuning experiments                                    |
| [**LlamaFactory**](./LlamaFactory/)                                                                                                                                                                                                            | LLM training and deployment framework                              |
| [**LLMArchitecture/ParameterCount**](./LLMArchitecture/ParameterCount/)                                                                                                                                                                        | Technical details of model architectures                           |
| [**Mistral-7b**](./Mistral-7b/)                                                                                                                                                                                                                | Mistral AI's 7B parameter model                                    |
| - `LLM_evaluation_harness_for_Arc_Easy_and_SST.ipynb`<br>- `Mistral_Colab_Finetune_ipynb_Colab_Final.ipynb`<br>- `notebooks_chatml_inference.ipynb`<br>- `notebooks_DPO_fine_tuning.ipynb`<br>- `notebooks_SFTTrainer TRL.ipynb`<br>- `SFT.py` | Comprehensive notebooks for evaluation, fine-tuning, and inference |
| [**Mixtral**](./Mixtral/)                                                                                                                                                                                                                      | Mixtral's mixture-of-experts model                                 |
| - `Mixtral_fine_tuning.ipynb`                                                                                                                                                                                                                  | Fine-tuning implementation                                         |
| [**VLM**](./VLM/)                                                                                                                                                                                                                              | Visual Language Models                                             |
| - `Florence2_finetuning_notebook.ipynb`<br>- `PaliGemma_finetuning_notebook.ipynb`                                                                                                                                                             | Implementations for vision-language models                         |

## 🎯 Module Overview

### 1. LLM Architectures

- Explore implementations of:
  - Llama2 (Meta's open-source model)
  - Mistral-7b (Efficient 7B parameter model)
  - Mixtral (Mixture-of-experts architecture)
  - Gemma (Google's latest contribution)
  - Llama3 (Upcoming experiments)

### 2. 🛠️ Fine-tuning Techniques

- Implementation strategies
- LoRA (Low-Rank Adaptation) approaches
- Advanced optimization methods

### 3. 🏗️ Model Architecture Analysis

- Deep dives into model structures
- Parameter counting methodologies
- Scaling considerations

### 4. 🔧 Specialized Implementations

- Code LLama for programming tasks
- Visual Language Models:
  - Florence2
  - PaliGemma

### 5. 💻 Practical Applications

- Comprehensive Jupyter notebooks
- Response generation pipelines
- Inference implementation guides

### 6. 🚀 Advanced Topics

- DPO (Direct Preference Optimization)
- SFT (Supervised Fine-Tuning)
- Evaluation methodologies

## 🤝 Contributing

We welcome contributions! See our contributing guidelines for more information.

## 📚 Resources

Each subdirectory contains detailed documentation and implementation guides. Check individual README files for specific instructions.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  Made with ❤️ for the AI community
</div> -->
