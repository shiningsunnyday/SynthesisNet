if [[ $1 -eq 1 ]]; then
    ncpu=1;
    batch_size=100;
else
    ncpu=100;
    batch_size=100;
fi

python scripts/reconstruct-targets.py \
    --skeleton-set-file results/viz/top_1000/skeletons-top-1000-train.pkl \
    --ckpt-rxn /home/msun415/SynTreeNet/results/logs/gnn//gnn/version_23/ckpts.epoch=395-val_accuracy_loss=0.30.ckpt \
    --ckpt-bb /home/msun415/SynTreeNet/results/logs/gnn//gnn/version_24/ckpts.epoch=44-val_nn_accuracy_loss=0.70.ckpt \
    --hash-dir results/hash_table-bb=1000-prods=2_new/ \
    --out-dir /home/msun415/SynTreeNet/results/viz/top_1000 \
    --top-k 1 \
    --test-correct-method preorder \
    --filter-only rxn bb \
    --top-bbs-file results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt \
    --ncpu $ncpu \
    --batch-size $batch_size \
    --forcing-eval \
    --mermaid        
