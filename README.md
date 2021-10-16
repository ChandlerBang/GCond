# GCond
A PyTorch implementation of paper ["Graph Condensation for Graph Neural Networks"](https://arxiv.org/abs/2110.07580)

Code will be released soon. Stay tuned :)


Abstract
----
We propose and study the problem of graph condensation for graph neural networks (GNNs). Specifically, we aim to condense the large, original graph into a small, synthetic and highly-informative graph, such that GNNs trained on the small graph and large graph have comparable performance. Extensive experiments have demonstrated the effectiveness of the proposed framework in condensing different graph datasets into informative smaller graphs. In particular, we are able to approximate the original test accuracy by 95.3% on Reddit, 99.8% on Flickr and 99.0% on Citeseer, while reducing their graph size by more than 99.9%, and the condensed graphs can be used to train various GNN architectures.


![]()

<div align=center><img src="https://github.com/ChandlerBang/GCond/blob/main/GCond.png" width="800"/></div>





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






