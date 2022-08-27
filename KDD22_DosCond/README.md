# DosCond

[KDD 2022] A PyTorch Implementation for ["Condensing Graphs via One-Step Gradient Matching"](https://arxiv.org/abs/2206.07746) under node classification setting. For graph classification setting, please refer to [https://github.com/amazon-research/DosCond](https://github.com/amazon-research/DosCond). 


Abstract
----
As training deep learning models on large dataset takes a lot of time and resources, it is desired to construct a small synthetic dataset with which we can train deep learning models sufficiently. There are recent works that have explored solutions on condensing image datasets through complex bi-level optimization. For instance, dataset condensation (DC) matches network gradients w.r.t. large-real data and small-synthetic data, where the network weights are optimized for multiple steps at each outer iteration. However, existing approaches have their inherent limitations: (1) they are not directly applicable to graphs where the data is discrete; and (2) the condensation process is computationally expensive due to the involved nested optimization. To bridge the gap, we investigate efficient dataset condensation tailored for graph datasets where we model the discrete graph structure as a probabilistic model. We further propose a one-step gradient matching scheme, which performs gradient matching for only one single step without training the network weights. 
