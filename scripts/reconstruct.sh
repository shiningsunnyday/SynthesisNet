if [[ $1 -eq 1 ]]; then
    python scripts/reconstruct-targets.py \
        --skeleton-set-file results/viz/top_1000/skeletons-top-1000-train.pkl \
        --ckpt-dir /home/msun415/SynTreeNet/results/logs/gnn_indv \
        --hash-dir results/hash_table-bb=1000-prods=2_new/ \
        --out-dir /home/msun415/SynTreeNet/results/viz/top_1000 \
        --top-k 1 \
        --forcing-eval \
        --test-correct-method reconstruct \
        --filter-only rxn bb \
        --top-bbs-file results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt \
        --ncpu 1 \
        --batch-size 10
else
    python scripts/reconstruct-targets.py \
        --skeleton-set-file results/viz/top_1000/skeletons-top-1000-train.pkl \
        --ckpt-dir /home/msun415/SynTreeNet/results/logs/gnn_indv \
        --hash-dir results/hash_table-bb=1000-prods=2_new/ \
        --out-dir /home/msun415/SynTreeNet/results/viz/top_1000 \
        --top-k 3 \
        --forcing-eval \
        --test-correct-method reconstruct \
        --filter-only rxn bb \
        --top-bbs-file results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt \
        --ncpu 100 \
        --batch-size 100
fi