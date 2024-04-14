export PYTHONPATH="/home/msun415/SynTreeNet/Single-Player-MCTS/"
export BUILDING_BLOCKS_FILE=data/assets/building-blocks/enamine_us_matched.csv ; 
export RXN_TEMPLATE_FILE=data/assets/reaction-templates/hb.txt ; 
export RXN_COLLECTION_FILE=data/assets/reaction-templates/reactions_hb.json.gz ; 
export EMBEDDINGS_KNN_FILE=data/assets/building-blocks/enamine_us_emb_fp_256.npy ; 
python scripts/train-rl.py \
    --surrogate results/logs/gnn/surrogates/600000_all/ckpts.epoch=69-val_nn_accuracy_loss=0.52.ckpt \
    --hash-dir results/hash_table-bb=1000-prods=2_new/ \
    --building_blocks_file $BUILDING_BLOCKS_FILE \
    --rxns_collection_file $RXN_COLLECTION_FILE \
    --embeddings_knn_file $EMBEDDINGS_KNN_FILE \
    --skeleton_class 1
# python scripts/rl.py \
#     --skeleton-set-file results/viz/top_1000/skeletons-top-1000-train.pkl \
#     --hash-dir results/hash_table-bb=1000-prods=2_new/ \
#     --out-dir /home/msun415/SynTreeNet/results/viz/top_1000 \
#     --top-k 1 \
#     --forcing-eval \
#     --test-correct-method reconstruct \
#     --filter-only rxn bb \
#     --top-bbs-file results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt \
#     --ncpu 1 \
