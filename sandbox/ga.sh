export PYTHONPATH="${HOME}/SynTreeNet/src"

python sandbox/optimize_ga.py \
    --skeleton_set_file /ssd/msun415/skeletons-top-1000-valid.pkl \
    --ckpt_rxn /ssd/msun415/surrogate/version_38/ \
    --ckpt_bb /ssd/msun415/surrogate/version_37/ \
    --filter_only rxn bb \
    --hash_dir /ssd/msun415/hash_table-bb=1000-prods=2_new/ \
    --top_bbs_file /ssd/msun415/bblocks-top-1000.txt \
    --objective drd2 \
    --out_dir $HOME/SynTreeNet/results/viz/top_1000 \
    --enable_wandb