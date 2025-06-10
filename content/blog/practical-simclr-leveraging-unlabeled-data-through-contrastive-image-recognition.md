+++
date = '2025-06-10T22:25:43+08:00'
draft = true
title = 'Practical SimCLR: Leveraging Unlabeled Data Through Contrastive Image Recognition'
+++

## Representation Learning

**All you need is a good representation**. Whether you're classifying images, grouping them into clusters, segmenting objects or detecting patterns, deep learning is fundamentally about teaching models to *learn meaningful representations of data*.

Good representation will lead to good results. For example, imagine you're looking at a group photo with your friends. You can instantly tell who's who because you've learned the "representation" of each friend. Let's say your friend Nyoman has a distinctive round face and curly hair while your friend Ketut is recognizable by her bright smile and glasses.

Your brain has unconsciously processed and stored these features, allowing you to recognize them effortlessly, even in a crowd or a different pose.

This mirrors deep learning's core objective: Transform raw data into compact representations where semantically similar items cluster together in feature space. For images, this means mapping an input image $I \in \mathbb{R}^{C \times H \times W}$ to a $D$-dimensional *latent* vector $z \in \mathbb{R}^D$:

$$ f: I \to \mathbb{R}^D $$

Since you already know the identity (label) of each of your friends, your brain will automatically infer the related features in the latent spaces and associate them with your friend's name. This process is often referred to as **supervised learning**.

Now, what if you're looking at a group photo of strangers you've never met? You might still be able to group them based on similar physical features, perhaps some have dark hair and are tall, while others are blonde and short. Here, *you don't know their identities or labels*.

In scenarios like this, where explicit human-provided labels are missing, we're broadly in the realm of **unsupervised learning**. This type of learning aims to discover hidden patterns or structures within the data itself, like simply clustering similar faces together without knowing their names. A really powerful approach within this area is **self-supervised learning**.

## Supervised learning vs unsupervised learning vs self-supervised learning

Based on [1], self-supervised learning (SSL) involves generating output labels “intrinsically” from input data examples by revealing the relationships between data components or various views of the data. These output labels are derived directly from the data examples. 

![Figure 1: The difference between supervised, unsupervised and self-supervised learning. Image source: [1]](/images/usl_vs_ssl.png)

**Supervised learning** operates with labeled pairs $\mathcal{D} = {(x_i, y_i)}{i=1}^N$ where $x \in \mathbb{R}^d$ is the cow image and $y \in \mathcal{Y}$ is the discrete label "cow". The objective is learning $f\theta: \mathbb{R}^d \rightarrow \mathcal{Y}$ by minimizing $\mathcal{L} = \mathbb{E}{(x,y) \sim \mathcal{D}}[\ell(f\theta(x), y)]$. As shown in Figure 1 (left), supervision $y$ comes externally: manual annotation provides ground truth labels.

**Unsupervised learning** works with unlabeled data $\mathcal{D} = {x_i}{i=1}^N$ where $x \in \mathbb{R}^d$. The goal is learning meaningful representations $g\phi: \mathbb{R}^d \rightarrow \mathbb{R}^k$ or discovering latent structure $p(x) = \int p(x|z)p(z)dz$. Figure 1 (middle) illustrates this: identical input $x$ but no external supervision: the model must discover patterns autonomously.

**Self-supervised learning** constructs *pseudo-labels* from data itself. Given $x$, we create $(x', y')$ pairs where $y' = h(x)$ for some transformation $h$. The loss becomes $\mathcal{L} = \mathbb{E}{x \sim \mathcal{D}}[\ell(f\theta(x'), h(x))]$. Figure 1 (right) demonstrates this perfectly: visual input $x_{visual}$ paired with co-occurring audio $x_{audio}$, where $y' = x_{audio}$ derives intrinsically from the multimodal data structure without external annotation.

## Families of self-supervised learning

Self-supervised learning can be done using 2 approaches: *Deep Metric Learning* and *Self Distillation* [2].

### Deep Metric Learning

Given an input $x \in \mathcal{X}$, we produce an augmentation or variant $\tilde{x}$ of $x$ through a semantic preserving transformation $T: \mathcal{X} \rightarrow \mathcal{X}$, where $\tilde{x} = T(x)$. This creates positive pairs $(x, \tilde{x})$ and negative pairs $(x, x')$ where $x' \neq x$. We then train a neural network $f_\theta: \mathcal{X} \rightarrow \mathbb{R}^d$ to minimize the contrastive loss $\mathcal{L}{contrastive}$, which makes the embeddings of positive pairs $f\theta(x)$ and $f_\theta(\tilde{x})$ closer in the representation space while pushing apart the embeddings of negative pairs $f_\theta(x)$ and $f_\theta(x')$. For the visual learner just like myself, Fig. 2 depict this process. 

![Figure 2: Deep Metric Learning framework](/images/deep-metric-learning.png)

One particular architecture for image classification that we will talk about in this post, SimCLR [3], uses Deep Metric Learning paradigm.

### Self Distillation

Given an input $x \in \mathcal{X}$, we generate two augmented views $x_1 = T_1(x)$ and $x_2 = T_2(x)$ using different data augmentation functions. The self-distillation paradigm employs two networks: an online network $f_\theta$ and a target network $f_\xi$, where $f_\theta, f_\xi: \mathcal{X} \rightarrow \mathbb{R}^d$. The target network parameters $\xi$ are updated as an exponential moving average of the online network parameters: $\xi \leftarrow \tau \xi + (1-\tau)\theta$, where $\tau \in [0,1]$ is the momentum coefficient.

Methods like BYOL (Bootstrap Your Own Latent) [4] train the online network to predict the target network's representation of $x_2$ given $x_1$, optimizing $\mathcal{L}{distill} = ||f\theta(x_1) - \text{sg}(f_\xi(x_2))||_2^2$, where $\text{sg}(\cdot)$ denotes the stop-gradient operation. These self-distillation methods will not be discussed further in this post.

## Similarity Contrastive Learning (SimCLR) for Image Classification

***tobecontinued***

## Citations:

[1] Gui, J., et al. "A Survey on Self-Supervised Learning: Algorithms, Applications, and Future Trends," in IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 12, pp. 9052–9071, 2024.

[2] Randall Balestriero, M., et al, "A Cookbook of Self-Supervised Learning," 2023.

[3] G. Ting Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations," 2020.

[4] Jean-Bastien Grill et al. "Bootstrap your own latent: A new approach to self-supervised Learning," 2020.



