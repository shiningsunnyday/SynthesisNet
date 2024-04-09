export BUILDING_BLOCKS_FILE=data/assets/building-blocks/enamine_us_matched.csv
export RXN_TEMPLATE_FILE=data/assets/reaction-templates/hb.txt
export RXN_COLLECTION_FILE=data/assets/reaction-templates/reactions_hb.json.gz
export EMBEDDINGS_KNN_FILE=data/assets/building-blocks/enamine_us_emb_fp_256.npy
export OMP_NUM_THREADS=1
if [[ $3 -eq 1 ]]; then
        metric=nn_accuracy_loss;
        loss=mse;
else
        metric=accuracy_loss;
        loss=cross_entropy;
fi
if [[ $1 -eq -1 ]]; then
        datasets='';
else
        datasets="--gnn-datasets $1";
fi
python src/synnet/models/gnn.py \
        --gnn-input-feats data/top_1000/gnn_featurized_leaf_up_2_split \
        --results-log /dccstor/graph-design/gnn/ \
        --mol-embedder-file $EMBEDDINGS_KNN_FILE \
        ${datasets} \
        --gnn-valid-loss ${metric} \
        --gnn-loss ${loss} \
        --gnn-layer Transformer \
        --lazy_load \
        --ncpu $2 \
        --prefetch_factor 0 \
        --feats-split \
        --cuda 0
