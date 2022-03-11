for r in 0.001 0.005 0.01
do
    bash scripts/script_cross.sh flickr ${r} 0  
    bash scripts/script_cross.sh ogbn-arxiv ${r} 0 
done

for r in 0.25 0.5 1
do 
    bash scripts/script_cross.sh citeseer ${r} 0  
    bash scripts/script_cross.sh cora ${r} 0  
done

for r in 0.001 0.0005 0.002
do     
    bash scripts/script_cross.sh reddit ${r} 0 
done
