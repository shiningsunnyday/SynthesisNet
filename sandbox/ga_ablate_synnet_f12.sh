export PYTHONPATH="${HOME}/SynTreeNet/src"
export LD_LIBRARY_PATH=/home/alston/miniforge3/envs/synnet/lib
MAX_NUM_RXNS=4

python sandbox/optimize.py \
    --seed=10 \
    --background_set_file /ssd/msun415/skeletons/skeletons-train.pkl \
    --skeleton_set_file /ssd/msun415/skeletons/skeletons-valid.pkl \
    --ckpt_rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 3 \
    --top_k_rxn 3 \
    --strategy conf \
    --objective $1 \
    --wandb=true \
    --wandb_project=syntreenet_ga_rebuttal_v3 \
    --method=ours \
    --num_workers=0 \
    --fp_bits=2048 \
    --bt_mutate_edits=3 \
    --early_stop=true \
    --early_stop_delta=0.01 \
    --early_stop_warmup=30 \
    --early_stop_patience=10 \
    --fp_mutate_prob=0.5 \
    --children_strategy "flips" \
    --max_oracle_workers=0 \
    --method=synnet --fp_bits=4096 --bt_ignore=true
