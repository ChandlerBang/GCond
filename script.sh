python train_cond_tranduct_sampler.py --dataset cora --mlp=0 --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.5  --seed=1000

python train_cond_tranduct_sampler.py --dataset ogbn-arxiv --mlp=0 --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  --lr_adj=0.01 --r=0.02   --seed=0 --inner=3  --epochs=1000  --save=0

