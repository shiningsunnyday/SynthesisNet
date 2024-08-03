# dataset=gnn_featurized_rxn_target_down_bb
# dataset=gnn_featurized_rxn_target_down_interm_postorder
dataset=gnn_featurized_leaves_up_postorder
for max_depth in {4,}; do
    for split in {'train','valid','test'}; do
        mkdir -p data/${dataset}_max_depth=${max_depth}_${split}
        python scripts/process-for-gnn.py \
            --determine_criteria leaves_up \
            --output-dir data/${dataset}_max_depth=${max_depth}_${split} \
            --anchor_type target \
            --visualize-dir results/viz/ \
            --skeleton-file results/viz/skeletons-${split}.pkl \
            --max_depth ${max_depth} \
            --ncpu 50 \
            --num-trees-per-batch 5000
        mkdir -p data/${dataset}_max_depth=${max_depth}_split_${split}
        python scripts/split_data.py \
            --in-dir data/${dataset}_max_depth=${max_depth}_${split}/ \
            --out-dir data/${dataset}_max_depth=${max_depth}_split_${split}/ \
            --partition_size 1        
    done;
done;

# for max_depth in {5,4,2,1}; do
#     for split in {'train','valid','test'}; do
#         python scripts/split_data.py \
#             --in-dir data/${dataset}_max_depth=${max_depth}_${split}/ \
#             --out-dir data/${dataset}_max_depth=${max_depth}_split_${split}/ \
#             --partition_size 1     
#     done;
# done;

# for split in {'train','valid','test'}; do
    # mkdir -p data/${dataset}_max_depth=${max_depth}_${split}
    # python scripts/process-for-gnn.py \
    #     --determine_criteria leaves_up \
    #     --output-dir data/${dataset}_max_depth=${max_depth}_${split} \
    #     --anchor_type target \
    #     --visualize-dir results/viz/ \
    #     --skeleton-file results/viz/skeletons-${split}.pkl \
    #     --max_depth 3 \
    #     --ncpu 100 \
    #     --num-trees-per-batch 5000
#     mkdir -p data/${dataset}_max_depth=${max_depth}_split_${split}
#     python scripts/split_data.py \
#         --in-dir data/${dataset}_max_depth=${max_depth}_${split}/ \
#         --out-dir data/${dataset}_max_depth=${max_depth}_split_${split}/ \
#         --partition_size 1
# done;




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
