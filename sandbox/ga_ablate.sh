export PYTHONPATH="${HOME}/SynTreeNet/src"
export OMP_NUM_THREADS=1
MAX_NUM_RXNS=4
MODEL_DIR=/ssd/msun415/surrogate
SKELETON_DIR=results/viz
python sandbox/optimize.py \
    --seed=$3 \
    --background_set_file ${SKELETON_DIR}/skeletons-train.pkl \
    --skeleton_set_file ${SKELETON_DIR}/skeletons-valid.pkl \
    --ckpt_rxn ${MODEL_DIR}/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb ${MODEL_DIR}/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer ${MODEL_DIR}/${MAX_NUM_RXNS}-REC/ \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 3 \
    --top_k_rxn 3 \
    --strategy conf \
    --objective $1 \
    --wandb=true \
    --wandb_project=syntreenet_ga_rebuttal_v3 \
    --method=ours \
    --num_workers=100 \
    --fp_bits=2048 \
    --bt_mutate_edits=3 \
    --early_stop=true \
    --early_stop_delta=0.01 \
    --early_stop_warmup=30 \
    --early_stop_patience=10 \
    --fp_mutate_prob=0.5 \
    --children_strategy "$2" \
    --max_oracle_workers=0
