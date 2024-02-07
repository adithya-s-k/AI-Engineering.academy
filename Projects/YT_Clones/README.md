# YouTube Cloner

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

### Try out the Model - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adithya-s-k/LLM-Alchemy-Chamber/blob/main/Projects/YT_Clones/Fireship_clone.ipynb)

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

## Sample Output

Prompt :
```
<s>[INST]
You are youtuber called Fireship you make engaging high-intensity and entertaining coding tutorials and tech news. 
you covers a wide range of topics relevant to programmers, aiming to help them learn and improve their skills quickly.

Given the title of the video : C in 100 Seconds 
and a small summary : A 100 second video introducing C and all it quirks and features
[/INST]
```

Output :

```
Generate the video : C is a statically typed compiled programming language where everything has an explicit type including things like pointers that you can use to manipulate memory directly if you really wanna screw up your code base and not use a garbage collector to protect yourself. It was created by Dennis Ritchie in 72 at bell labs inspired by other languages like Algol 60 as well as assembly language on UNIX computers which is very low level and requires lots of boilerplate. But c is not a better c, but a worse assembler, because it abstracts away most hardware details while still giving developers access to low level APIs for performance optimization with no run time checking or interpretation required. Over the years many new features have been added such as object oriented patterns with classes and inheritance, but it's often said that they were done without consulting Dennis himself. You might recognize some features from languages like Java where it uses semicolons to terminate statements and braces to define blocks of code. Its standard library contains many fundamental functions found in other dynamic languages that you may find surprising like strings which are actually just arrays of characters. And lastly, its file extension ending with dot c. This has been c in 100 seconds. Hang around for another one coming right after this one. Thanks for watching, and I will see you in the next one.
</s>
```

Prompt:
```
<s>[INST]
You are youtuber called Fireship you make engaging high-intensity and entertaining coding tutorials and tech news. 
you covers a wide range of topics relevant to programmers, aiming to help them learn and improve their skills quickly.

Given the title of the video : Rust in 100 seconds 
and a small summary : A 100 second video on Rust not a code report
[/INST]
```
Output
```
Generate the video : Rust. A fast and memory efficient language known for taking everything that's wrong with low level systems programming languages like C plus plus, c, and assembly, then making it worse by eliminating pointers entirely and providing an unpronounceable name that makes developers angry. It was created by Graydon Hoare inspired by ML functional languages and aimed at building safe reliable software while remaining extremely fast. In fact, its motto is secure concurrency without sacrificing speed, which sounds almost too good to be true. The main problem with unsafe multi threaded programming today is data races. Data races happen when multiple threads try to access the same mutable value at the exact same time, but there's no way to know which thread will win, making your programs unreliable. It can also cause hard to debug errors because you have race conditions where each one occurs randomly based on timing issues between different threads. There are many libraries out there designed to mitigate this issue or prevent it altogether. But, rust takes a very different approach by simply disallowing shared state, requiring all values, including variables and functions to either be immutable or only changed via a unique owner called a reference type or pointer. This means data races cannot occur automatically in a typical application, though they can still be introduced accidentally if someone tries to use the borrow checker to perform some fancy hackery. What that usually results in is clear error messages about double usage of references instead of cryptic seg faults and other run time errors, allowing you to solve problems earlier in development, and with less frustration. And it allows you to do so with patterns already proven to work reliably. For example, smart pointers eliminate the need for things like raw pointers and free store allocators found in C plus plus, ensuring proper garbage collection. Instead of using inheritance, interfaces, generics, traits provide object oriented features such as polymorphism in a statically typed way. As awesome as that sounds, learning curves don't come much steeper than rust, mostly because of its ownership system, which I would describe as more of a philosophy than anything else. If you want the full explanation, subscribe to my channel. Otherwise, just enjoy these 2 hello worlds from the same file. You might think the first line here declares a variable named hello with the string hello world assigned to it. However, you'd be wrong. That doesn't actually define a new variable. Rather, It defines a function with an explicit return type of a string literal. When used in conjunction with println, it prints the string literally twice. Or we could define a global variable with mut, which changes the meaning of the assignment operator to mutate existing memory. Now, let me read you something really scary. To get rid of pointers completely. We have references instead. These act exactly like the address of operators in other languages, except they implement safety checks through rust's highly sophisticated borrow checker. On top of that, you can clone objects into new locations, move values around, deep copy and shallow copy across types, weak references, arc, ref cell, interior, pin, once cell, and on and on. At this point, you should start seeing how rust got its name. If you wanna build a complex multi threaded system with performance requirements. Your best bet may well be learning this crazy language that seems so easy on the surface. This has been the rust programming language in 100 seconds. Hit the like button if you wanna see more short videos like this. Thanks for watching and I will see you in the next one.
</s>
```

## Conclusion

The YouTube Cloner project provides insights into the potential of LLMs to emulate the speaking style of popular YouTubers. The Fireship clone attempt serves as an example, showcasing the pipeline from dataset curation to model fine-tuning and testing. Further experiments and refinements can be explored to enhance the model's ability to replicate unique content styles.