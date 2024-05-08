export PYTHONPATH="${HOME}/SynTreeNet/src"
MAX_NUM_RXNS=4

python sandbox/optimize.py \
    --skeleton_set_file /ssd/msun415/skeletons/skeletons-valid.pkl \
    --ckpt_rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --max_rxns 0 \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 3 \
    --top_k_rxn 3 \
    --objective qed \
    --out_dir $HOME/SynTreeNet/results/viz/top_1000 \
    --enable_wandb \
    --method=ours \
    --num_workers=32 --chunksize=1 \
    --bt_nodes_max=10 --offspring_size=512
