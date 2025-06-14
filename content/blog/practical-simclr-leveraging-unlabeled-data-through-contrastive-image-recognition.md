---
title: "Practical Simclr Leveraging Unlabeled Data Through Contrastive Image Recognition"
date: 2025-06-15T00:23:10+08:00
readingTime: 5
categories: []
tags: []
# draft: true
---

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
### The Motivation

Previous works in unsupervised learning often involve a pretext task and the use of a memory bank to generate pseudolabels. A pretext task is an artificial learning objective designed to train a model without requiring human-annotated labels. Pseudolabels are automatically generated labels derived from the structure of the data or the task itself, rather than from manual labeling.

For example, the paper [5] uses image rotation as a pretext task. The model is trained to predict which geometric transformation (e.g., 0°, 90°, 180°, or 270° rotation) was applied to an input image. In this case, the rotation label serves as a pseudolabel, enabling the model to learn meaningful visual representations in a self-supervised manner.

Geometric transformations, such as image rotation, are chosen because a convolutional neural network cannot correctly predict the rotation of an image without understanding the salient object, local structures, and important features within the image [5]. If a model is able to accurately identify the type of rotation or geometric transformation applied, it suggests that the model has learned meaningful semantic representations. These representations are crucial for downstream tasks such as image classification, clustering, and segmentation.

However, having to manually set a lot of geometric transformations can limit generalization, as the model may overfit to those specific augmentations rather than learning broadly useful representations. This constraint arises because pretext tasks like rotation prediction rely on carefully designed transformations, which may not always capture the full complexity of real-world visual data.

If the chosen augmentations are too narrow or simplistic, the learned features may not transfer well to downstream tasks, while overly complex transformations could introduce noise or unnecessary learning challenges.

SimCLR circumvents this issue by replacing predefined pretext tasks with a contrastive learning framework, where the model learns by comparing different augmented views of the same image, allowing it to discover more flexible and generalizable visual features without relying on handcrafted transformation rules.

Another common approach in unsupervised learning is the use of *memory bank*, such as in MoCo (Momentum Contrast) [6]. The memory bank stored a dynamic dictionary of encoded features from previous batches, serving as negative samples for the current contrastive learning task.

While effective, this method has limitations:
1. The stored features can become stale, as they are not updated in real-time with the encoder, leading to inconsistency in the contrastive objective
2. Maintaining a large memory bank increases computational overhead and memory usage.

SimCLR addressed these issues by eliminating the memory bank entirely and instead using large batches with in-batch negatives. This approach ensured all compared features were encoded by the same up-to-date model while simplifying the training pipeline.

Although requiring more computational resources per batch, SimCLR's memory-free design produced superior representations by maintaining consistency across all contrastive comparisons.

The removal of the memory bank also made the framework more conceptually straightforward, as it relied solely on data augmentation and batch processing rather than maintaining an external feature storage system.

### How It Works

Given a mini-batch of images ${x_k}_{k=1}^N$ sampled from a dataset $\mathcal{D}$, we apply stochastic data augmentations twice to each image to obtain two correlated views $\tilde{x}_k^{(1)}$ and $\tilde{x}_k^{(2)}$:

$$
\tilde{x}_k^{(1)}, \tilde{x}_k^{(2)} \sim \mathcal{T}(x_k)
$$

where $\mathcal{T}$ is a family of random augmentation functions (e.g., cropping, color jitter). These augmented batches are passed through a shared encoder network $f_\theta$ to obtain representations:

$$
h_k^{(1)} = f_{\theta}(\tilde{x}_k^{(1)}) 
$$

$$ h_k^{(2)} = f_{\theta}(\tilde{x}_k^{(2)}) $$

These are further transformed by a projection head $g_\phi$ into the latent space:

$$
z_k^{(1)} = g_\phi(h_k^{(1)}), \quad z_k^{(2)} = g_\phi(h_k^{(2)})
$$


We then apply a contrastive loss $\mathcal{L}_{\text{contrastive}}$ over the latent pairs ${(z_k^{(1)}, z_k^{(2)})}$ to pull positive pairs together and push apart negatives. The training objective is to minimize the loss:

$$
\min_{\theta, \phi} \ \mathcal{L}_{\text{contrastive}}(z_k^{(1)}, z_k^{(2)})
$$

This encourages the model to learn meaningful representations by aligning the views of the same image while separating different images in the latent space.

While the math provides a compact explanation, the following code shows how it’s implemented in practice:

```python
train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCH}")
for epoch in range(MAX_EPOCH):
    simclr_model.train()
    # notice we don't use the label to train the model
    # normally, people often use something like `for batch_idx, image in enumerate(train_bar)`
    # or `for batch_idx, (image, _) in enumerate(train_bar)`
    # but in this example, the "label" is written just to demonstrate the common `dataloader.__getitem__()` return values format: (image, label)
    for batch_idx, (image, label) in enumerate(train_bar):
        optimizer.zero_grad()
        image.to(DEVICE)
        
        x_i, x_j = augmentor(image)

        # h is the encoded features
        # z is the latent projection
        hi, zi = simclr_model(x_i)
        hj, zj = simclr_model(x_j)

        loss = loss_fn(zi, zj)
        loss.backward()
        optimizer.step()
```
with the `augmentor` is defined as:
```python
# assuming the data in batch is a normalized tensor
class Augmentor:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    IMAGE_SIZE,
                    scale=(0.08, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
            ]
        )

    def __call__(self, imgbatch):
        aug1 = self.transform(imgbatch).to(DEVICE)
        aug2 = self.transform(imgbatch).to(DEVICE)
        return aug1, aug2

# instantiating the augmentor
augmentor = Augmentor()
```
As for the encoder $f_\theta$ and the projection head $g_\theta$, we have several options:

1. Use a pre-trained model (e.g., ResNet [7]) for $f_\theta$, or build a custom one from scratch.
2. Design $g_\theta$ as a simple neural network, typically a small MLP.

In this example, we use a pre-trained ResNet-18 as the feature encoder $f_\theta$, and define $g_\theta$ as a lightweight non-linear projection head:

```python
PRETRAINED_ENCODER = models.resnet18(weights=ResNet18_Weights.DEFAULT)
PRETRAINED_ENCODER.fc = torch.nn.Identity() # we are not going to use the fc

class Projector(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        hidden_features: int = 1024,
        out_features: int = 512,
    ):
        super(Projector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layers(x)


class SimClr(nn.Module):
    def __init__(self):
        super(SimClr, self).__init__()
        self.encoder = PRETRAINED_ENCODER
        self.projection = Projector(512, 512)

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        proj = self.projection(latent)

        return latent, proj
```
### NT-Xent Loss Explained

The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss is a contrastive objective function used in self-supervised learning. Given two normalized embedding vectors $z_i$ and $z_j$ from positive pairs (augmented views of the same image), the loss is computed as:

$$\mathcal{L} = -\frac{1}{N}\sum_{k=1}^N \log \frac{\exp(\text{sim}(z_i^{(k)}, z_j^{(k)}) / \tau)}{\sum_{m \neq k} \exp(\text{sim}(z_i^{(k)}, z_j^{(m)}) / \tau)}$$

where $\text{sim}(u,v) = u^Tv$ denotes cosine similarity, $\tau$ is the temperature parameter, $N$ is the batch size, and the denominator sums over all negative pairs (embeddings from different images).

**Step-by-Step Computation (Batch Size=2)**:  
Consider two images (A and B) with normalized embeddings:  
- $z_i = \begin{bmatrix} A1 \\ B1 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.6 \\ -0.7 & 0.7 \end{bmatrix}$  
- $z_j = \begin{bmatrix} A2 \\ B2 \end{bmatrix} = \begin{bmatrix} 0.9 & 0.4 \\ -0.8 & 0.6 \end{bmatrix}$  

<div class="ntxent-steps">
1. <b>Concatenate embeddings</b>  
   <div>$z = \begin{bmatrix} A1 \\ B1 \\ A2 \\ B2 \end{bmatrix} = \begin{bmatrix} 0.8 & 0.6 \\ -0.7 & 0.7 \\ 0.9 & 0.4 \\ -0.8 & 0.6 \end{bmatrix}$</div>
   <div class="why">Why: Creates a single tensor containing all embeddings from both views (A1/A2 and B1/B2) for efficient pairwise comparison</div>

2. **Similarity matrix** ($\tau=0.5$)  
   <div>$\texttt{sim} = \frac{z \cdot z^T}{0.5} = 2 \times \begin{bmatrix} 
   \color{gray}{1.0} & -0.14 & 0.96 & -0.28 \\ 
   -0.14 & \color{gray}{0.98} & -0.35 & 0.98 \\ 
   0.96 & -0.35 & \color{gray}{0.97} & -0.48 \\ 
   -0.28 & 0.98 & -0.48 & \color{gray}{1.0}
   \end{bmatrix} = \begin{bmatrix} 
   \color{gray}{2.0} & -0.28 & \color{green}{1.92} & -0.56 \\ 
   -0.28 & \color{gray}{1.96} & -0.70 & \color{green}{1.96} \\ 
   \color{green}{1.92} & -0.70 & \color{gray}{1.94} & -0.96 \\ 
   -0.56 & \color{green}{1.96} & -0.96 & \color{gray}{2.0}
   \end{bmatrix}$</div>
   <div class="why">Why: Measures similarity between all embedding pairs. Temperature $\tau$ sharpens the distribution</div>

3. **Mask diagonal**  
   <div>$\texttt{sim} = \begin{bmatrix} 
   \color{red}{\texttt{-1e9}} & -0.28 & \color{green}{1.92} & -0.56 \\ 
   -0.28 & \color{red}{\texttt{-1e9}} & -0.70 & \color{green}{1.96} \\ 
   \color{green}{1.92} & -0.70 & \color{red}{\texttt{-1e9}} & -0.96 \\ 
   -0.56 & \color{green}{1.96} & -0.96 & \color{red}{\texttt{-1e9}}
   \end{bmatrix}$</div>
   <div class="why">Why: Prevents trivial solution of self-similarity. We use <code>-1e9</code> (not -inf) for numerical stability</div>

4. **Create targets**  
   <div>$\texttt{targets} = \begin{bmatrix} 2 \\ 3 \\ 0 \\ 1 \end{bmatrix}$  
   (A1→A2 (index 2), B1→B2 (index 3), A2→A1 (index 0), B2→B1 (index 1))</div>
   <div class="why">Why: Maps each embedding to its positive pair's index for cross-entropy calculation</div>

5. **Cross-entropy** (for A1)  
   <div>$\texttt{logits} = [\texttt{-1e9}, -0.28, \color{green}{1.92}, -0.56]$  
   $\texttt{softmax} = [0, 0.03, \color{green}{0.95}, 0.02]$  
   $\texttt{loss} = -\log(\color{green}{0.95}) \approx 0.05$</div>
   <div class="why">Why: Maximizes similarity for positive pairs (green) while minimizing similarity to negatives</div>
</div>

The PyTorch implementation precisely follows this process with clear design choices:

```python
class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize to unit sphere (why: cosine similarity requires unit vectors)
        z_i = F.normalize(z_i, p=2, dim=1, eps=1e-6)
        z_j = F.normalize(z_j, p=2, dim=1, eps=1e-6)
        
        # Prevent extreme values (why: avoid NaN in gradient calculations)
        z_i = torch.clamp(z_i, -1, 1)
        z_j = torch.clamp(z_j, -1, 1)

        # Concatenate both views (why: enables batch-wise pairwise comparison)
        z = torch.cat([z_i, z_j], dim=0)  # (2*B, D)

        # Compute similarity matrix (why: measure all embedding relationships)
        sim = torch.matmul(z, z.T) / self.temperature  # (2B,2B)

        # Remove self-similarity (why: avoid trivial solution)
        N = 2 * batch_size
        diag_mask = torch.eye(N, device=device).bool()
        sim.masked_fill_(diag_mask, -1e9)  # Exact code value: -1e9

        # Create targets (why: define positive pairs for cross-entropy)
        # Structure: [positive_for_z_i, positive_for_z_j]
        targets = torch.arange(batch_size, device=device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        # Cross-entropy loss (why: standard objective for classification)
        loss = F.cross_entropy(sim, targets)
        
        # Handle NaN edge cases (why: training stability)
        if torch.isnan(loss):
            print("Warning: NaN loss detected!")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return loss
```

<style>
.ntxent-steps {
  margin: 20px 0;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #4e73df;
}
.ntxent-steps div {
  margin: 12px 0;
}
.ntxent-steps .why {
  font-size: 0.9em;
  padding: 8px 12px;
  margin-top: 5px;
  background: #e9ecef;
  border-radius: 4px;
  border-left: 3px solid #6c757d;
}
</style>

> Wait, there is a cross entropy in the NT-Xent formula? Where is it?  

The NT-Xent formula *is* a specialized cross entropy loss where:  
1. **Logits** are the similarity scores between an anchor embedding and all other embeddings  
2. **Target class** is the index of its positive pair  
3. **Temperature scaling** ($\tau$) sharpens the probability distribution  

The PyTorch implementation `F.cross_entropy(sim, targets)` computes:  
$$\mathcal{L} = -\frac{1}{2N}\sum_{k=0}^{2N-1} \log \frac{\exp(\text{sim}[k, \text{target}[k]])}{\sum_{j=0}^{2N-1} \exp(\text{sim}[k, j])}$$  

This exactly matches the NT-Xent formulation because:  
- The numerator $\exp(\text{sim}[k, \text{target}[k]])$ corresponds to $\exp(\text{sim}(z_i^{(k)}, z_j^{(k)})/\tau)$  
- The denominator sums over all embeddings (including negatives)  
- Diagonal masking ensures self-similarity ($j=k$) is excluded from the sum  

**Key insight**: The code implements NT-Xent as a multi-class classification task where each anchor must identify its positive pair among all other embeddings in the batch.


### Hyperparameters

The SimCLR paper emphasizes several critical hyperparameters for optimization:
- **Batch size**: Large batches (4096-8192) provide more negative samples, crucial for contrastive learning
- **Temperature** (τ): Typically set to 0.07, controls similarity distribution sharpness
- **Learning rate**: Base LR=0.3 with linear scaling (LR=0.075×batch_size/256)
- **Scheduler**: Cosine decay without restarts over long epochs (100-1000)
- **Projection head**: MLP with ReLU (hidden size=2048, output=128) improves representation
- **Augmentation strength**: Color distortion (strength=1.0) and Gaussian blur are essential

Training typically runs 100-1000 epochs using LARS optimizer with weight decay=1e-6 and gradient clipping.

### Limitations

SimCLR has several notable limitations:
1. **Computational cost**: Requires large batches (≥4096) and long training (100-1000 epochs)
2. **Augmentation sensitivity**: Performance heavily depends on carefully tuned augmentation strategies
3. **False negatives**: Treats different images of same class as negatives
4. **Projection head dependency**: Final representations require discarding the projection layer
5. **Batch uniformity**: Assumes all negatives are equally irrelevant
6. **Memory constraints**: Large batch sizes demand significant GPU/TPU resources

These limitations inspired subsequent approaches like MoCo (memory banks) [6] and BYOL (asymmetric networks)[4] to reduce computational demands.


### Downstream Network

The beauty of SimCLR lies in its versatility for downstream tasks. Once the encoder is pre-trained, you can attach **any classifier** to its output features for supervised learning. The key steps are:

1. Feature Extraction: Use the frozen SimCLR encoder to convert input images into rich latent representations
2. Classifier Design: Attach a simple neural network (e.g., MLP) that maps features to class labels
3. Transfer Learning: Only train the classifier while keeping the pre-trained encoder frozen

**Example Implementation**:
```python
# Simple MLP classifier
class DownstreamClassifier(nn.Module):
    def __init__(self, in_feat: int = 512, hidden_feat: int = 128, num_class: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feat, hidden_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_feat, hidden_feat),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_feat, num_class),
        )
        
    def forward(self, x):
        return self.layers(x)

# Training setup
downstream_model = DownstreamClassifier().to(DEVICE)
downstream_optim = torch.optim.Adam(downstream_model.parameters(), lr=1e-4)
downstream_loss_fn = nn.CrossEntropyLoss()

# Freeze SimCLR encoder
simclr_model.eval()
for param in simclr_model.parameters():
    param.requires_grad = False

# Training loop
for epoch in range(MAX_EPOCHS):
    for image, label in train_loader:
        with torch.no_grad():
            latent, _ = simclr_model(image)  # Extract features
            
        outputs = downstream_model(latent)    # Classify
        loss = downstream_loss_fn(outputs, label)
        loss.backward()
        downstream_optim.step()
```

In my implementation of contrastive learning using the SimCLR framework on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) dataset, I employ a ResNet18 encoder followed by a simple neural network for downstream classification, with the complete code available in this [Kaggle notebook](https://www.kaggle.com/code/teguhsatyadharma/ctrsl-wip).

Due to hardware constraints (SimCLR's substantial memory requirements causing resource exhaustion) and time limitations for finalizing this initial post, I conducted raw training without fine-tuning, achieving 71.50% validation accuracy.

While this result has room for optimization, it successfully demonstrates the core principles of contrastive learning in practice.

## Citations:

[1] Gui et al., A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends. 2024. [Online]. Available: https://arxiv.org/abs/2301.05712

[2] R. Balestriero et al., A Cookbook of Self-Supervised Learning. 2023. [Online]. Available: https://arxiv.org/abs/2304.12210

[3] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, A Simple Framework for Contrastive Learning of Visual Representations. 2020. [Online]. Available: https://arxiv.org/abs/2002.05709

[4] .-B. Grill et al., Bootstrap your own latent: A new approach to self-supervised Learning. 2020. [Online]. Available: https://arxiv.org/abs/2006.07733

[5] S. Gidaris, P. Singh, and N. Komodakis, Unsupervised Representation Learning by Predicting Image Rotations. 2018. [Online]. Available: https://arxiv.org/abs/1803.07728

[6] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, Momentum Contrast for Unsupervised Visual Representation Learning. 2020. [Online]. Available: https://arxiv.org/abs/1911.05722

[7] K. He, X. Zhang, S. Ren, and J. Sun, Deep Residual Learning for Image Recognition. 2015. [Online]. Available: https://arxiv.org/abs/1512.03385
