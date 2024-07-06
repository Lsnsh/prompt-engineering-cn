# What is a Large Language Model (LLM)?

A large language model is a prediction engine that takes a sequence of words
and tries to predict the most likely sequence to come after that sequence[^1].
It does this by assigning a probability to likely next sequences and then
samples from those to choose one[^2]. The process repeats until some stopping
criteria is met.

Large language models learn these probabilities by training on large corpuses
of text. A consequence of this is that the models will cater to some use cases
better than others (e.g. if it’s trained on GitHub data, it’ll understand the
probabilities of sequences in source code really well). Another consequence is
that the model may generate statements that seem plausible, but are actually
just random without being grounded in reality.

As language models become more accurate at predicting sequences, [many
surprising abilities
emerge](https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/).

[^1]: Language models actually use tokens, not words. A token roughly maps to a syllable in a word, or about 4 characters.
[^2]: There are many different pruning and sampling strategies to alter the behavior and performance of the sequences.

# A Brief, Incomplete, and Somewhat Incorrect History of Language Models

> :pushpin: Skip [to here](#what-is-a-prompt) if you'd like to jump past the
> history of language models. This section is for the curious minded, though
> may also help you understand the reasoning behind the advice that follows.

## Pre-2000’s

[Language models](https://en.wikipedia.org/wiki/Language_model#Model_types)
have existed for decades, though traditional language models (e.g. [n-gram
models](https://en.wikipedia.org/wiki/N-gram_language_model)) have many
deficiencies in terms of an explosion of state space ([the curse of
dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)) and
working with novel phrases that they’ve never seen (sparsity). Plainly, older
language models can generate text that vaguely resembles the statistics of
human generated text, but there is no consistency within the output – and a
reader will quickly realize it’s all gibberish. N-gram models also don’t scale
to large values of N, so are inherently limited.

## Mid-2000’s

In 2007, Geoffrey Hinton – famous for popularizing backpropagation in 1980’s –
[published an important advancement in training neural
networks](http://www.cs.toronto.edu/~fritz/absps/tics.pdf) that unlocked much
deeper networks. Applying these simple deep neural networks to language
modeling helped alleviate some of problems with language models – they
represented nuanced arbitrary concepts in a finite space and continuous way,
gracefully handling sequences not seen in the training corpus. These simple
neural networks learned the probabilities of their training corpus well, but
the output would statistically match the training data and generally not be
coherent relative to the input sequence. 

## Early-2010’s

Although they were first introduced in 1995, [Long Short-Term Memory (LSTM)
Networks](https://en.wikipedia.org/wiki/Long_short-term_memory) found their
time to shine in the 2010’s. LSTMs allowed models to process arbitrary length
sequences and, importantly, alter their internal state dynamically as they
processed the input to remember previous things they saw. This minor tweak led
to remarkable improvements. In 2015, Andrej Karpathy [famously wrote about
creating a character-level
lstm](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) that performed
far better than it had any right to.

LSTMs have seemingly magical abilities, but struggle with long term
dependencies. If you asked it to complete the sentence, “In France, we
traveled around, ate many pastries, drank lots of wine, ... lots more text ...
, but never learned how to speak _______”, the model might struggle with
predicting “French”. They also process input one token at a time, so are
inherently sequential, slow to train, and the `Nth` token only knows about the
`N - 1` tokens prior to it.

## Late-2010’s

In 2017, Google wrote a paper, [Attention Is All You
Need](https://arxiv.org/pdf/1706.03762.pdf), that introduced [Transformer
Networks](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
and kicked off a massive revolution in natural language processing. Overnight,
machines could suddenly do tasks like translating between languages nearly as
good as (sometimes better than) humans. Transformers are highly parallelizable
and introduce a mechanism, called “attention”, for the model to efficiently
place emphasis on specific parts of the input. Transformers analyze the entire
input all at once, in parallel, choosing which parts are most important and
influential. Every output token is influenced by every input token.

Transformers are highly parallelizable, efficient to train, and produce
astounding results. A downside to transformers is that they have a fixed input
and output size – the context window – and computation increases
quadratically with the size of this window (in some cases, memory does as
well!) [^3].

Transformers are not the end of the road, but the vast majority of recent
improvements in natural language processing have involved them. There is still
abundant active research on various ways of implementing and applying them,
such as [Amazon’s AlexaTM
20B](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning)
which outperforms GPT-3 in a number of tasks and is an order of magnitude
smaller in its number of parameters.

[^3]: There are more recent variations to make these more compute and memory efficient, but remains an active area of research.

## 2020’s

While technically starting in 2018, the theme of the 2020’s has been
Generative Pre-Trained models – more famously known as GPT. One
year after the “Attention Is All You Need” paper, OpenAI released [Improving
Language Understanding by Generative
Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).
This paper established that you can train a large language model on a massive
set of data without any specific agenda, and then once the model has learned
the general aspects of language, you can fine-tune it for specific tasks and
quickly get state-of-the-art results.

In 2020, OpenAI followed up with their GPT-3 paper [Language Models are
Few-Shot
Learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf),
showing that if you scale up GPT-like models by another factor of ~10x, in
terms of number of parameters and quantity of training data, you no
longer have to fine-tune it for many tasks. The capabilities emerge naturally
and you get state-of-the-art results via text interaction with the model.

In 2022, OpenAI followed-up on their GPT-3 accomplishments by releasing
[InstructGPT](https://openai.com/research/instruction-following). The intent
here was to tweak the model to follow instructions, while also being less
toxic and biased in its outputs. The key ingredient here was [Reinforcement
Learning from Human Feedback (RLHF)](https://arxiv.org/pdf/1706.03741.pdf), a
concept co-authored by Google and OpenAI in 2017[^4], which allows humans to
be in the training loop to fine-tune the model output to be more in line with
human preferences. InstructGPT is the predecessor to the now famous
[ChatGPT](https://en.wikipedia.org/wiki/ChatGPT).

OpenAI has been a major contributor to large language models over the last few
years, including the most recent introduction of
[GPT-4](https://cdn.openai.com/papers/gpt-4.pdf), but they are not alone. Meta
has introduced many open source large language models like
[OPT](https://huggingface.co/facebook/opt-66b),
[OPT-IML](https://huggingface.co/facebook/opt-iml-30b) (instruction tuned),
and [LLaMa](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).
Google released models like
[FLAN-T5](https://huggingface.co/google/flan-t5-xxl) and
[BERT](https://huggingface.co/bert-base-uncased). And there is a huge open
source research community releasing models like
[BLOOM](https://huggingface.co/bigscience/bloom) and
[StableLM](https://github.com/stability-AI/stableLM/).

Progress is now moving so swiftly that every few weeks the state-of-the-art is
changing or models that previously required clusters to run now run on
Raspberry PIs.

[^4]: 2017 was a big year for natural language processing.