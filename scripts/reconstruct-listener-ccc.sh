MAX_NUM_RXNS=3
TOP_K=3
TOP_K_RXN=3
STRATEGY=topological
MAX_RXNS=-1
use_case="analog_top_k=${TOP_K}_max_num_rxns=${MAX_NUM_RXNS}_max_rxns=${MAX_RXNS}_top_k_rxn=${TOP_K_RXN}_strategy=${STRATEGY}"
ROOT_DIR=/u/msun415/SynTreeNet/
MODEL_DIR=/u/msun415/SynTreeNet/surrogate/

python -u scripts/reconstruct_listener.py \
    --proc_id $1 \
    --filename input_${use_case}.txt \
    --output_filename output_${use_case}.txt \
    --skeleton-set-file results/viz/skeletons-valid.pkl \
    --ckpt-rxn ${MODEL_DIR}/${MAX_NUM_RXNS}-RXN/ \
    --ckpt-bb ${MODEL_DIR}/${MAX_NUM_RXNS}-NN/ \
    --ckpt-recognizer ${MODEL_DIR}/${MAX_NUM_RXNS}-REC/ \
    --out-dir ${ROOT_DIR}/results/viz/ \
    --top-k ${TOP_K} \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top-k-rxn ${TOP_K_RXN} \
    --max_rxns ${MAX_RXNS} \
    --test-correct-method reconstruct \
    --strategy ${STRATEGY} 