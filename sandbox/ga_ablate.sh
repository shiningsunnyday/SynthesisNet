export PYTHONPATH="${HOME}/SynTreeNet/src"
MAX_NUM_RXNS=4
export OMP_NUM_THREADS=1

python sandbox/optimize.py \
    --seed=10 \
    --background_set_file /ssd/msun415/skeletons/skeletons-train.pkl \
    --skeleton_set_file /ssd/msun415/skeletons/skeletons-valid.pkl \
    --ckpt_rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt_bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --ckpt_recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top_k 1 \
    --top_k_rxn 1 \
    --strategy topological \
    --objective $1 \
    --wandb \
    --wandb_project=syntreenet_ga_publish \
    --method=ours \
    --num_workers=50 \
    --offspring_size=512 \
    --analog_size=64 \
    --fp_bits=2048 \
    --bt_mutate_edits=2 \
    --early_stop \
    --early_stop_delta=0.01 \
    --early_stop_warmup=30 \
    --early_stop_patience=10 \
    # --initialize_path ./data/zinc.csv \
    # --resume_path ablations/population.pkl \
    # --generations=10 
    # --checkpoint_path ablations/population.pkl
    # --max_oracle_calls=5000 \