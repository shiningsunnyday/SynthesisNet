obj=analog
MAX_NUM_RXNS=3
TOP_K=3
TOP_K_RXN=3
MAX_RXNS=-1
STRATEGY=topological
use_case="mcmc_${obj}_top_k=${TOP_K}_top_k_rxn=${TOP_K_RXN}_max_rxns=${MAX_RXNS}_max_num_rxns=${MAX_NUM_RXNS}_strategy=${STRATEGY}"
batch_size=1000

python scripts/mcmc.py \
    --data data/assets/molecules/chembl_34_chemreps.tsv \
    --batch-size $batch_size \
    --skeleton-set-file results/viz/skeletons-valid.pkl \
    --ckpt-rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt-bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --out-dir /home/msun415/SynTreeNet/results/chembl/analog/ \
    --ckpt-recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --top-k ${TOP_K} \
    --top-k-rxn ${TOP_K_RXN} \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --max_rxns ${MAX_RXNS} \
    --test-correct-method reconstruct \
    --strategy ${STRATEGY} \
    --beta 10. \
    --mcmc_uniq 30 \
    --obj ${obj} \
    --sender-filename input_${use_case}.txt \
    --receiver-filename output_${use_case}.txt
    # --chunk_size 5 \
    # --ncpu $ncpu \   ddk 