+++
date = '2025-06-10T22:25:43+08:00'
draft = true
title = 'Practical SimCLR: Leveraging Unlabeled Data Through Contrastive Image Recognition'
+++

## Representation Learning

### Introduction

**All you need is a good representation**. Whether you're classifying images, grouping them into clusters, segmenting objects or detecting patterns, deep learning is fundamentally about teaching models to *learn meaningful representations of data*.

Good representation will lead to good results. For example, imagine you're looking at a group photo with your friends. You can instantly tell who's who because you've learned the "representation" of each friend. Let's say your friend Nyoman has a distinctive round face and curly hair while your friend Ketut is recognizable by her bright smile and glasses.

Your brain has unconsciously processed and stored these features, allowing you to recognize them effortlessly, even in a crowd or a different pose.

This mirrors deep learning's core objective: Transform raw data into compact representations where semantically similar items cluster together in feature space. For images, this means mapping an input image $I \in \mathbb{R}^{C \times H \times W}$ to a $D$-dimensional *latent* vector $z \in \mathbb{R}^D$:

$$ f: I \to \mathbb{R}^D $$


Since you already know the identity (label) of each of your friends, your brain will automatically infer the related features in the latent spaces and associate them with your friend's name. This process is often referred to as **supervised learning**.

Now, what if you're looking at a group photo of strangers you've never met? You might still be able to group them based on similar physical features, perhaps some have dark hair and are tall, while others are blonde and short. Here, *you don't know their identities or labels*.

In scenarios like this, where explicit human-provided labels are missing, we're broadly in the realm of **unsupervised learning**. This type of learning aims to discover hidden patterns or structures within the data itself, like simply clustering similar faces together without knowing their names. A really powerful approach within this area is **self-supervised learning**.

### Supervised learning vs unsupervised learning vs self-supervised learning

Based on [1], self-supervised learning (SSL) involves generating output labels “intrinsically” from input data examples by revealing the relationships between data components or various views of the data. These output labels are derived directly from the data examples. 

![The difference between supervised, unsupervised and self-supervised learning. Image source: [1]](/images/usl_vs_ssl.png)

**Supervised learning** operates with labeled pairs $\mathcal{D} = {(x_i, y_i)}{i=1}^N$ where $x \in \mathbb{R}^d$ is the cow image and $y \in \mathcal{Y}$ is the discrete label "cow". The objective is learning $f\theta: \mathbb{R}^d \rightarrow \mathcal{Y}$ by minimizing $\mathcal{L} = \mathbb{E}{(x,y) \sim \mathcal{D}}[\ell(f\theta(x), y)]$. As shown in Figure 1 (left), supervision $y$ comes externally: manual annotation provides ground truth labels.

**Unsupervised learning** works with unlabeled data $\mathcal{D} = {x_i}{i=1}^N$ where $x \in \mathbb{R}^d$. The goal is learning meaningful representations $g\phi: \mathbb{R}^d \rightarrow \mathbb{R}^k$ or discovering latent structure $p(x) = \int p(x|z)p(z)dz$. Figure 1 (middle) illustrates this: identical input $x$ but no external supervision: the model must discover patterns autonomously.

**Self-supervised learning** constructs *pseudo-labels* from data itself. Given $x$, we create $(x', y')$ pairs where $y' = h(x)$ for some transformation $h$. The loss becomes $\mathcal{L} = \mathbb{E}{x \sim \mathcal{D}}[\ell(f\theta(x'), h(x))]$. Figure 1 (right) demonstrates this perfectly: visual input $x_{visual}$ paired with co-occurring audio $x_{audio}$, where $y' = x_{audio}$ derives intrinsically from the multimodal data structure without external annotation.

***to be continued: contrastive learning as "the rising star" in self-supervised learning***

## Citations:

[1] Jie Gui, Tuo Chen, Jing Zhang, Qiong Cao, Zhenan Sun, Hao Luo, & Dacheng Tao. (2024). A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends.
