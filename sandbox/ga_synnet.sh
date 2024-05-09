export PYTHONPATH="${HOME}/SynTreeNet/src"
MAX_NUM_RXNS=4
CUDA_VISIBLE_DEVICES=-1

python sandbox/optimize.py \
    --background_set_file /ssd/msun415/skeletons/skeletons-train.pkl \
    --skeleton_set_file /ssd/msun415/skeletons/skeletons-valid.pkl \
    --ckpt_rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --max_rxns -1 \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 1 \
    --top_k_rxn 1 \
    --strategy conf \
    --objective jnk \
    --out_dir $HOME/SynTreeNet/results/viz/synnet \
    --wandb \
    --method=synnet \
    --num_workers=0 \
    --bt_nodes_max=10 --offspring_size=512 --fp_bits=4096
