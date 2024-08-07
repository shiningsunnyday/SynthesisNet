obj=drd2
export OMP_NUM_THREADS=16
MAX_NUM_RXNS=3
TOP_K=3
TOP_K_RXN=3
MAX_RXNS=-1
STRATEGY=conf
use_case="mcmc_${obj}_top_k=${TOP_K}_top_k_rxn=${TOP_K_RXN}_max_rxns=${MAX_RXNS}_max_num_rxns=${MAX_NUM_RXNS}_strategy=${STRATEGY}"
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

    python -u scripts/mcmc_listener.py \
        --proc_id $i \
        --skeleton-set-file results/viz/skeletons-valid.pkl \
        --ckpt-rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
        --ckpt-bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
        --out-dir /home/msun415/SynTreeNet/results/chembl/ \
        --ckpt-recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
        --top-k ${TOP_K} \
        --top-k-rxn ${TOP_K_RXN} \
        --max_num_rxns ${MAX_NUM_RXNS} \
        --max_rxns ${MAX_RXNS} \
        --test-correct-method reconstruct \
        --strategy ${STRATEGY} \
        --beta 10. 100. \
        --mcmc_timesteps 1000 \
        --obj ${obj} \
        --sender-filename input_${use_case}.txt \
        --receiver-filename output_${use_case}.txt &
done

