---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "提示词工程指南"
  tagline: "使用 OpenAI 的 GPT-4 等大型语言模型的提示和技巧。"
  actions:
    - theme: brand
      text: 指南
      link: /guide/what-is-a-large-language-model-llm

features:
  - title: 什么是大型语言模型（LLM）？
    details: 大型语言模型是一个预测引擎，它采用一个单词序列，并试图预测该序列之后最有可能出现的序列。它通过将概率分配给可能的下一个序列，然后从中采样来选择一个序列来实现这一点。该过程重复进行，直到满足某些停止标准。
  - title: 什么是提示词？
    details: 提示词，有时被称为上下文，是在模型开始生成输出之前提供给它的文本。它指导模型探索所学知识的特定领域，以便输出与您的目标相关。作为类比，如果您将语言模型视为源代码解释器，那么提示词就是要解释的源代码。
  - title: 为什么我们需要提示词工程？
    details: 提示词工程是编写提示词的艺术，让语言模型做我们希望它做的事情——就像软件工程是编写源代码让计算机做我们希望它们做的事情的艺术一样。
  - title: 策略
    details: 针对特定需求或问题的示例和策略。为了获得成功的提示词工程，您需要结合本文档中列举的所有策略的一些子集。不要害怕混合搭配——或者发明自己的方法。
---

