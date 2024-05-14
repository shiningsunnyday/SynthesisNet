#!~/anaconda3/envs/synnet/bin/python
export PYTHONPATH="${HOME}/SynTreeNet/src"
MAX_NUM_RXNS=$3
export OMP_NUM_THREADS=1
MODEL_DIR=/dccstor/graph-design/surrogate
SKELETON_DIR=results/viz

python sandbox/optimize.py \
    --background_set_file ${SKELETON_DIR}/skeletons-train.pkl \
    --skeleton_set_file ${SKELETON_DIR}/skeletons-valid.pkl \
    --ckpt_rxn ${MODEL_DIR}/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb ${MODEL_DIR}/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer ${MODEL_DIR}/${MAX_NUM_RXNS}-REC/ \
    --max_rxns -1 \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 1 \
    --top_k_rxn 1 \
    --strategy conf \
    --objective $1 \
    --wandb \
    --method=ours \
    --num_workers=10 \
    --bt_nodes_max=25 \
    --offspring_size=512 \
    --fp_bits=2048 \
    --bt_crossover=recognizer \
    --bt_mutate_edits=$2
