# GCond
[ICLR 2022] A PyTorch implementation of paper ["Graph Condensation for Graph Neural Networks"](https://arxiv.org/abs/2110.07580)



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
For reddit, flickr and arxiv, we use the datasets provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT). 
They are available on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg)). Rename the folder to `data` at the root directory. Note that the links are provided by GraphSAINT team. 

## Run the code
For transductive setting, please run the following command:
```
python train_gcond_transduct.py --dataset cora --nlayers=2 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  
```
where `r` indicates the ratio of condensed samples to the labeled samples. For instance, there are only 140 labeled nodes in Cora dataset, so `r=0.5` indicates the number of condensed samples are 70.  

For inductive setting, please run the following command:
```
python train_gcond_induct.py --dataset flickr --nlayers=2 --lr_feat=0.01 --gpu_id=0  --lr_adj=0.01 --r=0.005 --epochs=1000  --outer=10 --inner=1
```

## Reproduce the performance
For Table 2, run `bash scripts/run_main.sh`.

For Table 3, run `bash scripts/run_cross.sh`.


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





