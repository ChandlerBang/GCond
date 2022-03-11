dataset=${1}
r=${2}
gpu_id=${3}
for s in 0 1 2 3 4
do
python test_other_arcs.py --dataset ${dataset} --gpu_id=${gpu_id} --r=${r} --seed=${s} --nruns=10  >> res/flickr/${1}_${2}.out 
done
