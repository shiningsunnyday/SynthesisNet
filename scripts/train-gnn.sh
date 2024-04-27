export BUILDING_BLOCKS_FILE=data/assets/building-blocks/enamine_us_matched.csv
export RXN_TEMPLATE_FILE=data/assets/reaction-templates/hb.txt
export RXN_COLLECTION_FILE=data/assets/reaction-templates/reactions_hb.json.gz
export EMBEDDINGS_KNN_FILE=data/assets/building-blocks/enamine_us_emb_fp_256.npy
export OMP_NUM_THREADS=1

debug=0;

# HPARAMS
ncpu=50;
# rewire='--rewire-edges';
rewire='';
pe='--pe sin';
# datasets='--gnn-datasets 1'
datasets=''
dataset=gnn_featurized_rxn_target_down_bb_postorder_split

# datasets='';
if [[ $1 -eq 1 ]]; then
        metric=nn_accuracy_loss;
        loss=mse;
else
        metric=accuracy_loss;
        loss=cross_entropy;
fi
if [[ ${debug} -eq 0 ]]; then
    python src/synnet/models/gnn.py \
            --gnn-input-feats data/top_1000/$dataset \
            --results-log results/logs/gnn/ \
            --mol-embedder-file $EMBEDDINGS_KNN_FILE \
            --gnn-valid-loss ${metric} \
            ${datasets} \
            --gnn-loss ${loss} \
            ${rewire} \
            ${pe} \
            --gnn-layer Transformer \
            --lazy_load \
            --ncpu ${ncpu} \
            --prefetch_factor 2 \
            --feats-split \
            --cuda 0
else
        python src/synnet/models/gnn.py \
            --gnn-input-feats data/top_1000/$dataset \
            --results-log results/logs/gnn/ \
            --mol-embedder-file $EMBEDDINGS_KNN_FILE \
            --gnn-valid-loss ${metric} \
            ${datasets} \
            --gnn-loss ${loss} \
            ${rewire} \
            ${pe} \
            --gnn-layer Transformer \
            --lazy_load \
            --ncpu 0 \
            --prefetch_factor 2 \
            --feats-split \
            --cuda 0
fi
