export PYTHONPATH="${HOME}/SynthesisNet/src"
export LD_LIBRARY_PATH=/home/alston/miniforge3/envs/synnet/lib
MAX_NUM_RXNS=4
export OMP_NUM_THREADS=1

PMO_ORACLES=('gsk')

for oracle in "${PMO_ORACLES[@]}"
do
  python sandbox/optimize.py \
      --seed=10 \
      --background_set_file /ssd/msun415/skeletons/skeletons-train.pkl \
      --skeleton_set_file /ssd/msun415/skeletons/skeletons-valid.pkl \
      --ckpt_rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN-EP/ \
      --ckpt_bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN-EP/ \
      --ckpt_recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
      --max_num_rxns ${MAX_NUM_RXNS} \
      --top_k 1 \
      --top_k_rxn 1 \
      --strategy topological \
      --max_topological_orders 5 \
      --objective ${oracle} \
      --wandb=true \
      --wandb_project=alston_syntreenet_ga_rebuttal_iclr \
      --method=ours \
      --num_workers=30 \
      --fp_bits=2048 \
      --bt_mutate_edits=3 \
      --early_stop=true \
      --early_stop_delta=0.01 \
      --early_stop_warmup=30 \
      --early_stop_patience=10 \
      --fp_mutate_prob=0.5 \
      --children_strategy edits \
      --max_oracle_workers=0
done