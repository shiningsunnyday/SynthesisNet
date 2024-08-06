export PYTHONPATH="${HOME}/SynTreeNet/src"
export LD_LIBRARY_PATH=/home/alston/miniforge3/envs/synnet/lib
MAX_NUM_RXNS=4
export OMP_NUM_THREADS=1

python sandbox/benchmark.py \
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
    --max_topological_orders 1 \
    --objective qed \
    --wandb=true \
    --wandb_project=alston_syntreenet_ga_benchmark \
    --num_workers=0 \
    --max_oracle_workers=0 \
    --bt_mutate_edits=3 \
    --children_strategy "edits" \
    --generations=-1 \
    --population=1000 \
    --method=ours --fp_bits=2048 \
    --method=synnet --fp_bits=4096 --bt_ignore=true \
