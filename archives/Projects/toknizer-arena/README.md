## Tokenizer Arena

The Tokenizer Arena project allows you to easily compare different tokenizers used by various LLMs. It serves the following purposes:

- Visualizing how a tokenizer tokenizes a given piece of text.
- Comparing it easily with other LLM tokenizers.

A good example of why I built this is a few days ago when LLama3 was released, people wanted to know how effective it is in tokenizing Indic languages. I had to download each tokenizer locally and write Python scripts to do so. Here is a tweet I wrote about it: [Tweet Link](https://x.com/adithya_s_k/status/1781018407519043720).

With Tokenizer Arena, all you have to do is type or paste a text, and you can see how each tokenizer tokenizes it and how efficient it is. For example, if you want to see how Kannada is tokenized, you can simply type in Kannada text