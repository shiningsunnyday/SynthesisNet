python scripts/ga-surrogate.py --log_file /home/msun415/SynTreeNet/results/viz/top_1000/log_ga.txt \
    --ckpt-rxn /home/msun415/SynTreeNet/results/logs/gnn//gnn/version_38/ \
    --ckpt-bb /home/msun415/SynTreeNet/results/logs/gnn//gnn/version_37/ \
    --filter-only rxn bb \
    --hash-dir results/hash_table-bb=1000-prods=2_new/ \
    --objective qed \
    --config_file /home/msun415/SynTreeNet/ga-config.json