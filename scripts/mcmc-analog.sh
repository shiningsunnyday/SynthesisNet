MAX_NUM_RXNS=3 # surrogate to use
TOP_K=3 # num beams for bb
TOP_K_RXN=3 # num beams for rxn
MAX_RXNS=-1 # use -1 for mcmc
STRATEGY=topological # decoding order
batch_size=1000 # logging purpose

MODEL_DIR=/dccstor/graph-design/surrogate # replace with yours
use_case="mcmc_${obj}_top_k=${TOP_K}_top_k_rxn=${TOP_K_RXN}_max_rxns=${MAX_RXNS}_max_num_rxns=${MAX_NUM_RXNS}_strategy=${STRATEGY}"

python scripts/mcmc.py \
    --data data/assets/molecules/chembl_34_chemreps.tsv \
    --batch-size $batch_size \
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
    --receiver-filename output_${use_case}.txt
