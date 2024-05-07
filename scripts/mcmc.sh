use_case='mcmc_top_k=3_max_num_rxns=3.txt'
batch_size=1000

python scripts/mcmc.py \
    --data data/assets/molecules/chembl_34_chemreps.tsv \
    --batch-size $batch_size \
    --skeleton-set-file results/viz/skeletons-valid.pkl \
    --ckpt-rxn /ssd/msun415/surrogate/version_42/ \
    --ckpt-bb /ssd/msun415/surrogate/version_70/ \
    --out-dir /home/msun415/SynTreeNet/results/chembl/ \
    --ckpt-recognizer /ssd/msun415/recognizer/ckpts.epoch=1-val_loss=0.14.ckpt \
    --top-k 3 \
    --max_num_rxns 3 \
    --max_rxns -1 \
    --test-correct-method reconstruct \
    --strategy topological \
    --beta 1. \
    --mcmc_timesteps 100 \
    --sender-filename input_${use_case}.txt \
    --receiver-filename output_${use_case}.txt \
    # --chunk_size 5 \
    # --ncpu $ncpu \    