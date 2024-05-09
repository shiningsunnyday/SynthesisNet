MAX_NUM_RXNS=3
use_case="mcmc_top_k=3_max_num_rxns=${MAX_NUM_RXNS}.txt"
batch_size=1000

python scripts/mcmc.py \
    --data data/assets/molecules/chembl_34_chemreps.tsv \
    --batch-size $batch_size \
    --skeleton-set-file results/viz/skeletons-valid.pkl \
    --ckpt-rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt-bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --out-dir /home/msun415/SynTreeNet/results/chembl/ \
    --ckpt-recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --top-k 3 \
    --top-k-rxn 3 \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --max_rxns -1 \
    --test-correct-method reconstruct \
    --strategy topological \
    --beta 1. \
    --mcmc_timesteps 20 \
    --sender-filename input_${use_case}.txt \
    --receiver-filename output_${use_case}.tx
    # --chunk_size 5 \
    # --ncpu $ncpu \    