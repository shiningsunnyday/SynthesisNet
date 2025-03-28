MAX_NUM_RXNS=4
TOP_K=3
TOP_K_RXN=3
STRATEGY=topological
MAX_RXNS=-1
use_case="analog_top_k=${TOP_K}_max_num_rxns=${MAX_NUM_RXNS}_max_rxns=${MAX_RXNS}_top_k_rxn=${TOP_K_RXN}_strategy=${STRATEGY}"
ncpu=1;
batch_size=1000;
ROOT_DIR=${HOME}/SynthesisNet/
# MODEL_DIR=${HOME}/SynthesisNet/surrogate/
MODEL_DIR=/ssd/msun415/surrogate

# python scripts/reconstruct-targets.py \
#     --skeleton-set-file results/viz/top_1000/skeletons-top-1000-valid.pkl \
#     --ckpt-rxn /ssd/msun415/surrogate/version_38/ \
#     --ckpt-bb /ssd/msun415/surrogate/version_37/ \
#     --ckpt-recognizer /ssd/msun415/recognizer/ckpts.epoch=3-val_loss=0.15.ckpt \
#     --hash-dir results/hash_table-bb=1000-prods=2_new/ \
#     --out-dir SynthesisNet/results/viz/top_1000 \
#     --top-k 3 \
#     --test-correct-method reconstruct \
#     --strategy topological \
#     --filter-only rxn bb \
#     --top-bbs-file results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt \
#     --ncpu $ncpu \
#     --batch-size $batch_size \
#     --sender-filename input_reconstruct.txt \
#     --receiver-filename output_reconstruct.txt

# python scripts/reconstruct-targets.py \
#     --skeleton-set-file results/viz/skeletons-valid.pkl \
#     --ckpt-rxn /ssd/msun415/surrogate/version_42/ \
#     --ckpt-bb /ssd/msun415/surrogate/version_70/ \
#     --ckpt-recognizer /ssd/msun415/recognizer/ckpts.epoch=1-val_loss=0.14.ckpt \
#     --out-dir SynthesisNet/results/viz/ \
#     --top-k 3 \
#     --test-correct-method reconstruct \
#     --ncpu $ncpu \
#     --batch-size $batch_size \
#     --mermaid \
#     --one-per-class

python scripts/reconstruct-targets.py \
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
    --ncpu $ncpu \
    --batch-size $batch_size \
    --num-analogs 5 \
    --sender-filename input_${use_case}.txt \
    --receiver-filename output_${use_case}.txt
