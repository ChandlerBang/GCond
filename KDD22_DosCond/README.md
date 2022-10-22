# DosCond

[KDD 2022] A PyTorch Implementation for ["Condensing Graphs via One-Step Gradient Matching"](https://arxiv.org/abs/2206.07746) under node classification setting. For graph classification setting, please refer to [https://github.com/amazon-research/DosCond](https://github.com/amazon-research/DosCond). 


Abstract
----
As training deep learning models on large dataset takes a lot of time and resources, it is desired to construct a small synthetic dataset with which we can train deep learning models sufficiently. There are recent works that have explored solutions on condensing image datasets through complex bi-level optimization. For instance, dataset condensation (DC) matches network gradients w.r.t. large-real data and small-synthetic data, where the network weights are optimized for multiple steps at each outer iteration. However, existing approaches have their inherent limitations: (1) they are not directly applicable to graphs where the data is discrete; and (2) the condensation process is computationally expensive due to the involved nested optimization. To bridge the gap, we investigate efficient dataset condensation tailored for graph datasets where we model the discrete graph structure as a probabilistic model. We further propose a one-step gradient matching scheme, which performs gradient matching for only one single step without training the network weights. 


Here we do not implement the discrete structure learning, but only borrow the idea from ["Condensing Graphs via One-Step Gradient Matching"](https://arxiv.org/abs/2206.07746) to perform one-step gradient matching, which significantly speeds up the condensation process.


Essentially, we can run the following commands:
```
python train_gcond_transduct.py --dataset citeseer --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=0.5  --sgc=0 --dis=mse --one_step=1  --epochs=3000
python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=0.5  --sgc=0 --dis=mse --gpu_id=2 --one_step=1  --epochs=5000
python train_gcond_transduct.py --dataset pubmed --nlayers=2 --lr_feat=1e-3 --lr_adj=1e-3 --r=0.5  --sgc=0 --dis=mse --gpu_id=2 --one_step=1  --epochs=2000
python train_gcond_transduct.py --dataset ogbn-arxiv --nlayers=2 --lr_feat=1e-2 --lr_adj=2e-2 --r=0.001  --sgc=1 --dis=ours --gpu_id=2 --one_step=1  --epochs=1000
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=5e-3 --lr_adj=5e-3 --r=0.001  --sgc=0 --dis=mse --gpu_id=3 --one_step=1  --epochs=1000
```
Note that using smaller learning rate and larger epochs can get even higher performance.


## Cite
For more information, you can take a look at the [paper](https://arxiv.org/abs/2206.07746).

If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{jin2022condensing,
  title={Condensing Graphs via One-Step Gradient Matching},
  author={Jin, Wei and Tang, Xianfeng and Jiang, Haoming and Li, Zheng and Zhang, Danqing and Tang, Jiliang and Yin, Bing},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={720--730},
  year={2022}
}
```

