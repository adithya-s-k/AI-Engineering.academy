# YouTube Cloner Project

## Introduction

The YouTube Cloner project aims to explore the capabilities of Language Models (LLMs) in capturing the speaking style of popular YouTubers. The primary objective is to understand how well LLMs can emulate the tone, pacing, and content style of specific YouTube channels by fine-tuning them on curated datasets.

The project focuses on the following key steps:

1. **Dataset Curation:**
   - Scrape video links from target YouTube channels.
   - Download videos, extract audio, and obtain video transcripts.
   - Summarize transcripts and create a dataset containing video titles, transcripts, and summaries.

2. **Model Fine-Tuning:**
   - Fine-tune LLMs on the curated dataset to capture the speaking style of the target YouTubers.

3. **Testing and Evaluation:**
   - Use the fine-tuned models to generate content based on provided prompts.
   - Evaluate the models' ability to replicate the style and content of the original YouTubers.

<!-- ## General Pipeline

1. **Dataset Curation:**
   - Use the `youtube_channel_scraper.py` script to scrape video links.
   - Follow the steps outlined in the `dataset_prep.ipynb` notebook for detailed dataset preparation, including audio extraction, transcript generation, and dataset formatting.


2. **Model Fine-Tuning:**
   - Fully fine-tune Llama2 on the dataset ([Hugging Face Model](https://huggingface.co/CognitiveLab/Fireship-clone-hf)).
   - Additionally, fine-tune a MIstral 7b version using Axolotl (yet to be announced).

3. **Testing the Model:**
   - Use the provided notebook to test the generation of content.
   - Follow the specified prompt template for instructing the model.
 -->
## Attempt 1: Fireship Clone

### Try out the Model

### Fireship (Original YouTuber)
- Channel: [Fireship](https://www.youtube.com/c/fireship)
- Style: Informative and fast-paced coding tutorials and tech news.



### Procedure:
1. **Link Extraction:**
   - Use `youtube_channel_scraper.py` to extract all video links.

2. **Dataset Preparation:**
   - Refer to the `dataset_prep.ipynb` notebook for detailed steps in preparing the dataset, including audio extraction, transcript generation, and dataset formatting.
   - Dataset available on [Hugging Face Datasets](https://huggingface.co/datasets/AdithyaSK/Fireship_transcript_summar_prompt).

3. **Model Fine-Tuning:**
   - Llama2 fully fine-tuned on the Fireship dataset ([Hugging Face Model](https://huggingface.co/CognitiveLab/Fireship-clone-hf)).
   - MIstral 7b version fine-tuned using Axolotl (to be announced).

4. **Testing the Generation:**
   - Utilize the provided notebook for testing content generation.
   - Follow the specified prompt template for instructing the model.

#### Prompt Template:
```
[INST]
You are a YouTuber called Fireship, creating engaging, high-intensity coding tutorials and tech news. 
You cover a wide range of topics relevant to programmers, aiming to help them learn and improve their skills quickly.

Given the title of the video: {title} 
and a small summary: {video_summary}
[/INST]

Generate the video: 
```

## Conclusion

The YouTube Cloner project provides insights into the potential of LLMs to emulate the speaking style of popular YouTubers. The Fireship clone attempt serves as an example, showcasing the pipeline from dataset curation to model fine-tuning and testing. Further experiments and refinements can be explored to enhance the model's ability to replicate unique content styles.