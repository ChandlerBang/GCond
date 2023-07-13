# GCond
[ICLR 2022] The PyTorch implementation for ["Graph Condensation for Graph Neural Networks"](https://cse.msu.edu/~jinwei2/files/GCond.pdf) is provided under the main directory. 

[KDD 2022] The implementation for ["Condensing Graphs via One-Step Gradient Matching"](https://arxiv.org/abs/2206.07746) is shown in the `KDD22_DosCond` directory. See [link](https://github.com/ChandlerBang/GCond/tree/main/KDD22_DosCond).


Abstract
----
We propose and study the problem of graph condensation for graph neural networks (GNNs). Specifically, we aim to condense the large, original graph into a small, synthetic and highly-informative graph, such that GNNs trained on the small graph and large graph have comparable performance. Extensive experiments have demonstrated the effectiveness of the proposed framework in condensing different graph datasets into informative smaller graphs. In particular, we are able to approximate the original test accuracy by 95.3% on Reddit, 99.8% on Flickr and 99.0% on Citeseer, while reducing their graph size by more than 99.9%, and the condensed graphs can be used to train various GNN architectures.


![]()

<div align=center><img src="https://github.com/ChandlerBang/GCond/blob/main/GCond.png" width="800"/></div>


## Requirements
Please see [requirements.txt](https://github.com/ChandlerBang/GCond/blob/main/requirements.txt).
```
torch==1.7.0
torch_geometric==1.6.3
scipy==1.6.2
numpy==1.19.2
ogb==1.3.0
tqdm==4.59.0
torch_sparse==0.6.9
deeprobust==0.2.4
scikit_learn==1.0.2
```

## Download Datasets
For cora, citeseer and pubmed, the code will directly download them; so no extra script is needed.
For reddit, flickr and arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). 
They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg)). Rename the folder to `data` at the root directory. Note that the links are provided by GraphSAINT team. 




## Run the code
For transductive setting, please run the following command:
```
python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  
```
where `r` indicates the ratio of condensed samples to the labeled samples. For instance, there are only 140 labeled nodes in Cora dataset, so `r=0.5` indicates the number of condensed samples are 70, **which corresponds to  r=2.6%=70/2710 in our paper**. Thus, the parameter `r` is different from the real reduction rate in the paper for the transductive setting, please see the following table for the correspondence.

|              | `r` in the code     | `r` in the paper  (real reduction rate)    |
|--------------|-------------------|---------------------|
| Transductive | Cora, r=0.5       | Cora, r=2.6%        |
| Transductive | Citeseer, r=0.5   | Citeseer, r=1.8%    |
| Transductive | Ogbn-arxiv, r=0.5 | Ogbn-arxiv, r=0.25% |
| Transductive | Pubmed, r=0.5     | Pubmed, r=0.3%      |
| Inductive    | Flickr, r=0.01    | Flickr, r=1%        |
| Inductive    | Reddit, r=0.001   | Reddit, r=0.1%      |

For inductive setting, please run the following command:
```
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=0.01 --gpu_id=0  --lr_adj=0.01 --r=0.005 --epochs=1000  --outer=10 --inner=1
```

## Reproduce the performance
The generated graphs are saved in the folder `saved_ours`; you can directly load them to test the performance.

For Table 2, run `bash scripts/run_main.sh`.

For Table 3, run `bash scripts/run_cross.sh`.

## [Faster Condensation!] One-Step Gradient Matching
From the KDD'22 paper ["Condensing Graphs via One-Step Gradient Matching"](https://arxiv.org/abs/2206.07746), we know that performing gradient matching for only one step can also achieve a good performance while significantly accelerating the condensation process. Hence, we can run the following command to perform one-step gradient matching, which is essentially much faster than the original version:
```
python train_gcond_transduct.py --dataset citeseer --nlayers=2 --lr_feat=1e-2 --lr_adj=1e-2 --r=0.5 \
    --sgc=0 --dis=mse --gpu_id=2 --one_step=1  --epochs=3000
```
For more commands, please go to [`KDD22_DosCond`](https://github.com/ChandlerBang/GCond/tree/main/KDD22_DosCond).

**[Note]: I found that sometimes using MSE loss for gradient matching can be more stable than using `ours` loss**, and it gives more flexibility on the model used in condensation (using GCN as the backbone can also generate good condensed graphs). 


## Whole Dataset Performance 
When we do coreset selection, we need to first the model on the whole dataset. Thus we can obtain the performanceo of whole dataset by running `train_coreset.py` and `train_coreset_induct.py`:
```
python train_coreset.py --dataset cora --r=0.01  --method=random
python train_coreset_induct.py --dataset flickr --r=0.01  --method=random
```

## Coreset Performance
Run the following code to get the coreset performance for transductive setting.
```
python train_coreset.py --dataset cora --r=0.01  --method=herding
python train_coreset.py --dataset cora --r=0.01  --method=random
python train_coreset.py --dataset cora --r=0.01  --method=kcenter
```
Similarly, run the following code for the inductive setting.
```
python train_coreset_induct.py --dataset flickr --r=0.01  --method=kcenter
```


## Cite
If you find this repo to be useful, please cite our two papers. Thank you!
```
@inproceedings{
    jin2022graph,
    title={Graph Condensation for Graph Neural Networks},
    author={Wei Jin and Lingxiao Zhao and Shichang Zhang and Yozen Liu and Jiliang Tang and Neil Shah},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=WLEx3Jo4QaB}
}
```

```
@inproceedings{jin2022condensing,
  title={Condensing Graphs via One-Step Gradient Matching},
  author={Jin, Wei and Tang, Xianfeng and Jiang, Haoming and Li, Zheng and Zhang, Danqing and Tang, Jiliang and Yin, Bing},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={720--730},
  year={2022}
}
```

