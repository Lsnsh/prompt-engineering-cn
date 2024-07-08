# 什么是大型语言模型（LLM）？

大型语言模型是一个预测引擎，它采用一个单词序列，并试图预测该序列之后最有可能出现的序列（语言模型实际上使用 Token ，而不是单词。一个 Token 大致映射到一个单词中的一个音节，或大约 4 个字符。）。

它通过将概率分配给可能的下一个序列，然后从中采样来选择一个序列（有许多不同的修剪和采样策略来改变序列的行为和性能。）来实现这一点。该过程重复进行，直到满足某些停止标准。

大型语言模型通过在大型文本语料库上进行训练来学习这些概率。其结果是，这些模型将比其他模型更好地适应某些用例（例如，如果它是在 GitHub 数据上训练的，它将非常好地理解源代码中序列的概率）。另一个后果是，该模型可能会生成看似合理的陈述，但实际上只是随机的，没有现实依据。

随着语言模型在预测序列方面变得更加准确，出现了[许多令人惊讶的能力](https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/)。

# 语言模型的简史、不完整史和某些不正确史

> :pushpin: 如果你想跳过语言模型的历史，请跳[到这里](/guide/what-is-a-prompt)。
> 这一部分是为好奇的人准备的，不过也可能帮助你理解以下建议背后的原因。

## 2000 年以前

[语言模型](https://en.wikipedia.org/wiki/Language_model#Model_types)已经存在了几十年，尽管传统的语言模型（如 [n-gram](https://en.wikipedia.org/wiki/N-gram_language_model) 模型）在状态空间的爆炸（[维度的诅咒](https://en.wikipedia.org/wiki/Curse_of_dimensionality)）和使用他们从未见过的新颖短语（稀疏性）方面存在许多缺陷。很明显，旧的语言模型可以生成与人类生成的文本的统计数据有点相似的文本，但输出中没有一致性 - 读者很快就会意识到这都是胡言乱语。N-gram 模型也不能扩展到大的 N 值，因此本质上是有限的。

## 2000 年代中期

2007 年，因在 20 世纪 80 年代推广反向传播而闻名的 - 杰弗里·辛顿 - [发表了一项训练神经网络的重要进展](http://www.cs.toronto.edu/~fritz/absps/tics.pdf)，该进展解锁了更深层次的网络。将这些简单的深度神经网络应用于语言建模有助于缓解语言模型的一些问题 —— 它们以有限的空间和连续的方式表示细微的任意概念，优雅地处理训练语料库中没有的序列。这些简单的神经网络很好地学习了其训练语料库的概率，但输出在统计上与训练数据匹配，并且通常相对于输入序列不一致。

## 2010 年初

尽管它们于 1995 年首次引入，但[长短期记忆（LSTM）网络](https://en.wikipedia.org/wiki/Long_short-term_memory)在 2010 年代大放异彩。LSTM 允许模型处理任意长度的序列，重要的是，在处理输入时动态改变其内部状态，以记住以前看到的东西。这个小小的调整带来了显著的改进。2015 年，Andrej Karpathy [写了一篇著名的文章，讲述了创造一个角色级的 lstm](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)，它的表现远远好于任何权利。

LSTM 具有看似神奇的能力，但却与长期依赖作斗争。如果你让它完成这句话，“在法国，我们四处旅行，吃了很多糕点，喝了很多酒，……还有很多文字……但从未学会说 **\_\_\_**”，该模型可能很难预测“法语”。它们还一次处理一个 Token 的输入，因此本质上是顺序的，训练缓慢，并且第 `N` 个 Token 只知道之前的 `N-1` 个 Token。

## 2010 年末

2017 年，谷歌写了一篇论文[《注意力就是你所需要的一切》](https://arxiv.org/pdf/1706.03762.pdf)，介绍了 [Transformer 网络](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>)，并开启了自然语言处理的一场巨大革命。一夜之间，机器可以突然完成语言之间的翻译等任务，几乎和人类一样好（有时甚至比人类更好）。转换器具有高度并行性，并引入了一种称为“注意力”的机制，使模型能够有效地将重点放在输入的特定部分。变压器同时并行分析整个输入，选择哪些部分最重要、最有影响力。每个输出 Token 都受到每个输入 Token 的影响。

变压器具有高度并行性，训练效率高，并产生惊人的结果。transformer 的一个缺点是，它们具有固定的输入和输出大小（上下文窗口），并且计算量随着该窗口的大小而二次增加（在某些情况下，内存也会增加！）—— 最近有一些变化可以提高计算和内存效率，但仍然是一个活跃的研究领域。

变形金刚并不是道路的尽头，但最近自然语言处理的绝大多数改进都涉及到了变形金刚。对于实现和应用它们的各种方式，仍有大量积极的研究，例如[亚马逊的 AlexaTM 20B](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning)，它在许多任务中优于 GPT-3，并且在参数数量上要小一个数量级。

## 2020 年

从技术上讲，从 2018 年开始，2020 年的主题一直是生成预训练模型——更著名的是 GPT。在“注意力就是你所需要的”论文发表一年后，OpenAI 发布了[《通过生成预训练提高语言理解》](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)。这篇论文证明，你可以在没有任何特定议程的情况下，在大量数据集上训练一个大型语言模型，然后一旦模型学习了语言的一般方面，你就可以针对特定任务对其进行微调，并迅速获得最先进的结果。

2020 年，OpenAI 跟进了他们的 GPT-3 论文[《语言模型是少数射击学习者》](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)，表明如果你将类似 GPT 的模型再放大约 10 倍，就参数数量和训练数据量而言，你就不必再为许多任务微调它了。这些功能自然出现，您可以通过与模型的文本交互获得最先进的结果。

2022 年，OpenAI 发布了 [InstructGPT](https://openai.com/research/instruction-following)，以跟进他们的 GPT-3 成就。这里的目的是调整模型以遵循指示，同时减少其毒性和输出的偏见。这里的关键成分是[从人类反馈中强化学习（RLHF）](https://arxiv.org/pdf/1706.03741.pdf)，这是谷歌和 OpenAI 于 2017 年共同撰写的一个概念（2017 年是自然语言处理的重要一年。），它允许人类进入训练循环，对模型输出进行微调，使其更符合人类的偏好。InstructionGPT 是现在著名的 [ChatGPT](https://en.wikipedia.org/wiki/ChatGPT) 的前身。

在过去几年里，OpenAI 一直是大型语言模型的主要贡献者，包括最近引入的 [GPT-4](https://cdn.openai.com/papers/gpt-4.pdf)，但它们并不是唯一一个。Meta 引入了许多开源的大型语言模型，如 [OPT](https://huggingface.co/facebook/opt-66b)、[OPT-IML](https://huggingface.co/facebook/opt-iml-30b)（指令调优）和 [LLaMa](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)。谷歌发布了 [FLAN-T5](https://huggingface.co/google/flan-t5-xxl) 和 [BERT](https://huggingface.co/bert-base-uncased) 等机型。还有一个庞大的开源研究社区，发布 [BLOOM](https://huggingface.co/bigscience/bloom) 和 [StableLM](https://github.com/stability-AI/stableLM/) 等模型。

现在进展如此之快，以至于每隔几周最先进的技术就会发生变化，或者以前需要集群运行的模型现在在 Raspberry PIs 上运行。
