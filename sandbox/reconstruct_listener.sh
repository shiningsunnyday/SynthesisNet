MAX_NUM_RXNS=6
export PYTHONPATH="${HOME}/SynTreeNet/src"
export OMP_NUM_THREADS=1
use_case="ga"
for ((i =1; i <= $1; i++));
do
    python -u scripts/reconstruct_listener.py \
        --proc_id $i \
        --filename input_${use_case}.txt \
        --output_filename output_${use_case}.txt \
        --skeleton-set-file /ssd/msun415/skeletons/skeletons-valid.pkl \
        --ckpt-rxn /ssd/msun415/surrogate/${MAX_NUM_RXNS}-RXN/ \
        --ckpt-bb /ssd/msun415/surrogate/${MAX_NUM_RXNS}-NN/ \
        --ckpt-recognizer /ssd/msun415/surrogate/${MAX_NUM_RXNS}-REC/ \
        --out-dir /home/msun415/SynTreeNet/results/viz/ \
        --top-k 3 \
        --max_num_rxns ${MAX_NUM_RXNS} \
        --top-k-rxn 3 \
        --max_rxns -1 \
        --test-correct-method reconstruct \
        --strategy conf &
done

