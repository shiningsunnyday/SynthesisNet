#!~/anaconda3/envs/synnet/bin/python
export PYTHONPATH="${HOME}/SynTreeNet/src"
MAX_NUM_RXNS=$3
MODEL_DIR=/dccstor/graph-design/surrogate
SKELETON_DIR=results/viz

python sandbox/optimize.py \
    --seed=10 \
    --background_set_file ${SKELETON_DIR}/skeletons-train.pkl \
    --skeleton_set_file ${SKELETON_DIR}/skeletons-valid.pkl \
    --ckpt_rxn ${MODEL_DIR}/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb ${MODEL_DIR}/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer ${MODEL_DIR}/${MAX_NUM_RXNS}-REC/ \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 1 \
    --top_k_rxn 1 \
    --strategy topological \
    --objective $1 \
    --wandb \
    --method=ours \
    --num_workers=0 \
    --offspring_size=256 \
    --analog_size=256 \
    --fp_bits=2048 \
    --bt_mutate_edits=$2
