export PYTHONPATH="${HOME}/SynTreeNet/src"

python sandbox/optimize_ga.py \
    --skeleton-set-file /ssd/msun415/skeletons-top-1000-valid.pkl \
    --ckpt-rxn /ssd/msun415/surrogate/version_38/ \
    --ckpt-bb /ssd/msun415/surrogate/version_37/ \
    --filter-only rxn bb \
    --hash-dir /ssd/msun415/hash_table-bb=1000-prods=2_new/ \
    --top-bbs-file /ssd/msun415/bblocks-top-1000.txt \
    --objective drd2 \
    --out-dir $HOME/SynTreeNet/results/viz/top_1000 \
    --enable_wandb