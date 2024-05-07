MAX_NUM_RXNS=6
export OMP_NUM_THREADS=1
use_case="reconstruct_top_k=3_max_num_rxns=${MAX_NUM_RXNS}_max_rxns=-1"
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
        --filename input_${use_case}.txt \
        --output_filename output_${use_case}.txt \
        --skeleton-set-file results/viz/skeletons-valid.pkl \
        --ckpt-rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
        --ckpt-bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
        --ckpt-recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
        --out-dir /home/msun415/SynTreeNet/results/viz/ \
        --top-k 3 \
        --max_num_rxns ${MAX_NUM_RXNS} \
        --top-k-rxn 3 \
        --max_rxns -1 \
        --test-correct-method reconstruct \
        --strategy topological &    
done

