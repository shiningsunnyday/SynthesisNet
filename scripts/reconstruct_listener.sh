MAX_NUM_RXNS=4
TOP_K=3
TOP_K_RXN=3
STRATEGY=topological
MAX_RXNS=-1
use_case="analog_top_k=${TOP_K}_max_num_rxns=${MAX_NUM_RXNS}_max_rxns=${MAX_RXNS}_top_k_rxn=${TOP_K_RXN}_strategy=${STRATEGY}"
ncpu=1
ROOT_DIR=${HOME}/SynTreeNet/
# MODEL_DIR=${HOME}/SynTreeNet/surrogate/
MODEL_DIR=/ssd/msun415/surrogate
export OMP_NUM_THREADS=1

# MODEL_DIR=/ssd/msun415/surrogate

for ((i =1; i <= $1; i++));
do
# python -u scripts/reconstruct_listener.py \
#     --proc_id $i \
#     --filename input_${use_case}.txt \
#     --output_filename output_${use_case}.txt \
#     --skeleton-set-file results/viz/top_1000/skeletons-top-1000.pkl \
#     --ckpt-rxn /ssd/msun415/surrogate/version_38/ \
#     --ckpt-bb /ssd/msun415/surrogate/version_37/ \
#     --ckpt-recognizer /ssd/msun415/recognizer/ckpts.epoch=3-val_loss=0.15.ckpt \
#     --hash-dir results/hash_table-bb=1000-prods=2_new/ \
#     --out-dir ${HOME}/SynTreeNet/results/viz/top_1000 \
#     --top-k 3 \
#     --test-correct-method reconstruct \
#     --strategy topological \
#     --filter-only rxn bb \
#     --top-bbs-file results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt &

    # python -u scripts/reconstruct_listener.py \
    #     --proc_id $i \
    #     --filename input_${use_case}.txt \
    #     --output_filename output_${use_case}.txt \
    #     --skeleton-set-file results/viz/skeletons.pkl \
    #     --ckpt-rxn /ssd/msun415/surrogate/version_42/ \
    #     --ckpt-bb /ssd/msun415/surrogate/version_70/ \
    #     --ckpt-recognizer /ssd/msun415/recognizer/ckpts.epoch=1-val_loss=0.14.ckpt \
    #     --out-dir ${HOME}/SynTreeNet/results/viz/ \
    #     --top-k 3 \
    #     --test-correct-method reconstruct \
    #     --strategy topological &

    python -u scripts/reconstruct_listener.py \
        --proc_id $i \
        --skeleton-set-file results/viz/skeletons-train.pkl \
        --ckpt-rxn ${MODEL_DIR}/${MAX_NUM_RXNS}-RXN \
        --ckpt-bb ${MODEL_DIR}/${MAX_NUM_RXNS}-NN \
        --ckpt-recognizer ${MODEL_DIR}/${MAX_NUM_RXNS}-REC/ \
        --out-dir ${ROOT_DIR}/results/viz/ \
        --top-k ${TOP_K} \
        --max_num_rxns ${MAX_NUM_RXNS} \
        --top-k-rxn ${TOP_K_RXN} \
        --max_rxns ${MAX_RXNS} \
        --test-correct-method reconstruct \
        --strategy ${STRATEGY} \
        --filename input_${use_case}.txt \
        --output_filename output_${use_case}.txt &
done
