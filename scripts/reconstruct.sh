MAX_NUM_RXNS=6

use_case="reconstruct_top_k=3_max_num_rxns=${MAX_NUM_RXNS}_max_rxns=-1"
ncpu=100;
batch_size=100000;

# python scripts/reconstruct-targets.py \
#     --skeleton-set-file results/viz/top_1000/skeletons-top-1000-valid.pkl \
#     --ckpt-rxn /ssd/msun415/surrogate/version_38/ \
#     --ckpt-bb /ssd/msun415/surrogate/version_37/ \
#     --ckpt-recognizer /ssd/msun415/recognizer/ckpts.epoch=3-val_loss=0.15.ckpt \
#     --hash-dir results/hash_table-bb=1000-prods=2_new/ \
#     --out-dir /home/msun415/SynTreeNet/results/viz/top_1000 \
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
#     --out-dir /home/msun415/SynTreeNet/results/viz/ \
#     --top-k 3 \
#     --test-correct-method reconstruct \
#     --ncpu $ncpu \
#     --batch-size $batch_size \
#     --mermaid \
#     --one-per-class \
#     --attn_weights
    # --sender-filename input_reconstruct.txt \
    # --receiver-filename output_reconstruct.txt

python scripts/reconstruct-targets.py \
    --skeleton-set-file results/viz/skeletons-valid.pkl \
    --ckpt-rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
    --ckpt-bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
    --out-dir /home/msun415/SynTreeNet/results/viz/ \
    --top-k 3 \
    --max_num_rxns ${MAX_NUM_RXNS} \
    --top-k-rxn 3 \
    --max_num_rxns 3 \
    --max_rxns -1 \
    --test-correct-method reconstruct \
    --strategy topological \
    --ncpu $ncpu \
    --batch-size $batch_size \
    --ckpt-recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
    --sender-filename input_${use_case}.txt \
    --receiver-filename output_${use_case}.txt
