for r in 0.25 0.5 1
do
python train_cond_tranduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=1 --epoch=600 --save=0
done

for r in 0.25 0.5 1
do
python train_cond_tranduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=${r}  --seed=1 --epoch=600 --save=0
done


for r in 0.001 0.005 0.01
do
python train_cond_tranduct.py --dataset ogbn-arxiv --nlayers=2 --sgc=1 --lr_feat=0.01 --gpu_id=3  --lr_adj=0.01 --r=${r}  --seed=1 --inner=3  --epochs=1000  --save=0
done


for r in 0.001 0.005 0.01
do
    python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.005  --r=${r} --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 
done

for r in 0.001 0.005 0.0005
do
    python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=${r} --seed=1 --gpu_id=0 --epochs=1000  --inner=1 --outer=10 --save=0 
done
