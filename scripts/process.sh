python scripts/process-for-gnn.py \
    --determine_criteria rxn_target_down_bb \
    --output-dir data/top_1000/gnn_featurized_rxn_target_down_bb_${split} \
    --anchor_type target \
    --skeleton-file results/viz/top_1000/skeletons-top-1000-${split}.pkl \
    --ncpu 50 \
    --num-trees-per-batch 50

# for split in {'valid','test'}; do
#     mkdir -p data/top_1000/gnn_featurized_leaf_up_2_${split}
#     python scripts/process-for-gnn.py \
#         --determine_criteria leaf_up_2 \
#         --output-dir data/top_1000/gnn_featurized_leaf_up_2_${split} \
#         --anchor_type target \
#         --skeleton-file results/viz/top_1000/skeletons-top-1000-${split}.pkl \
#         --ncpu 50 \
#         --num-trees-per-batch 5000
# done;

# for split in {'train','valid','test'}; do
#     mkdir -p data/top_1000/gnn_featurized_leaf_up_2_split_${split}
#     python scripts/split_data.py \
#         --in-dir data/top_1000/gnn_featurized_leaf_up_2_${split}/ \
#         --out-dir data/top_1000/gnn_featurized_leaf_up_2_split_${split}/ \
#         --partition_size 1
# done;
