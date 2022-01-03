# GCond
A PyTorch implementation of paper ["Graph Condensation for Graph Neural Networks"](https://arxiv.org/abs/2110.07580)



Abstract
----
We propose and study the problem of graph condensation for graph neural networks (GNNs). Specifically, we aim to condense the large, original graph into a small, synthetic and highly-informative graph, such that GNNs trained on the small graph and large graph have comparable performance. Extensive experiments have demonstrated the effectiveness of the proposed framework in condensing different graph datasets into informative smaller graphs. In particular, we are able to approximate the original test accuracy by 95.3% on Reddit, 99.8% on Flickr and 99.0% on Citeseer, while reducing their graph size by more than 99.9%, and the condensed graphs can be used to train various GNN architectures.


![]()

<div align=center><img src="https://github.com/ChandlerBang/GCond/blob/main/GCond.png" width="800"/></div>


## Some Notes
The current code provides the essential implemention of GCond. Currently it contains some repetitive code blocks and comments. A more detailed and clean version will be updated soon. 


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
For reddit, flickr and arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). Please follow their instruction to download. 

## Run the code
----
For transductive setting, please run the following command:
```
python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  
```
where `r` indicates the ratio of condensed samples to the labeled samples. For instance, there are only 140 labeled nodes in Cora dataset, so `r=0.5` indicates the number of condensed samples are 70.  

For inductive setting, please run the following command:
```
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=0.01 --gpu_id=0  --lr_adj=0.01 --r=0.005 --epochs=1000  --outer=20 
```


## Cite
For more information, you can take a look at the [paper](https://arxiv.org/abs/2110.07580).

If you find this repo to be useful, please cite our paper. Thank you.
```
@misc{jin2021graph,
      title={Graph Condensation for Graph Neural Networks}, 
      author={Wei Jin and Lingxiao Zhao and Shichang Zhang and Yozen Liu and Jiliang Tang and Neil Shah},
      year={2021},
      eprint={2110.07580},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```






