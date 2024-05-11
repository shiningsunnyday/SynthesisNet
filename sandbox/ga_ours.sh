export PYTHONPATH="${HOME}/SynTreeNet/src"
MAX_NUM_RXNS=6
export OMP_NUM_THREADS=1

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
    --objective $1 \
    --wandb \
    --method=ours \
    --num_workers=10 \
    --bt_nodes_max=10 \
    --offspring_size=512 \
    --fp_bits=2048
