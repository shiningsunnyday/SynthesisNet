obj=analog
MAX_NUM_RXNS=3
TOP_K=3
TOP_K_RXN=3
MAX_RXNS=-1
STRATEGY=topological

use_case="mcmc_${obj}_top_k=${TOP_K}_top_k_rxn=${TOP_K_RXN}_max_rxns=${MAX_RXNS}_max_num_rxns=${MAX_NUM_RXNS}_strategy=${STRATEGY}"
MODEL_DIR=/dccstor/graph-design/surrogate

export OMP_NUM_THREADS=1

for ((i =1; i <= $1; i++));
do
    python -u scripts/mcmc_listener.py \
        --proc_id $i \
        --skeleton-set-file results/viz/skeletons-valid.pkl \
        --ckpt-rxn ${MODEL_DIR}/${MAX_NUM_RXNS}-RXN/ \
        --ckpt-bb ${MODEL_DIR}/${MAX_NUM_RXNS}-NN/ \
        --out-dir results/chembl/analog/ \
        --ckpt-recognizer ${MODEL_DIR}/${MAX_NUM_RXNS}-REC/ \
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
        --receiver-filename output_${use_case}.txt &
done

