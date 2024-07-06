---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Prompt Engineering Chinese Translation"
  # text: "Tips and tricks for working with Large Language Models like OpenAI's GPT-4."
  tagline: "Tips and tricks for working with Large Language Models like OpenAI's GPT-4."
  actions:
    - theme: brand
      text: Prompt Engineering
      link: /guide/what-is-a-large-language-model-llm

features:
  - title: What is a Large Language Model (LLM)?
    details: A large language model is a prediction engine that takes a sequence of words and tries to predict the most likely sequence to come after that sequence[^1]. It does this by assigning a probability to likely next sequences and then samples from those to choose one[^2]. The process repeats until some stopping criteria is met.
  - title: What is a prompt?
    details: A prompt, sometimes referred to as context, is the text provided to a model before it begins generating output. It guides the model to explore a particular area of what it has learned so that the output is relevant to your goals. As an analogy, if you think of the language model as a source code interpreter, then a prompt is the source code to be interpreted.
  - title: Why do we need prompt engineering?
    details: Up above, we used an analogy of prompts as the “source code” that a language model “interprets”. **Prompt engineering is the art of writing prompts to get the language model to do what we want it to do** – just like software engineering is the art of writing source code to get computers to do what we want them to do.
  - title: Strategies
    details: This section contains examples and strategies for specific needs or problems. For successful prompt engineering, you will need to combine some subset of all of the strategies enumerated in this document. Don’t be afraid to mix and match things – or invent your own approaches.
---

