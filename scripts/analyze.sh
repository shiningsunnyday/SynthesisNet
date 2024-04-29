# python scripts/analyze-skeletons.py \
#     --skeleton-file results/viz/top_1000/skeletons-top-1000.pkl \
#     --input-file data/pre-process/syntrees/synthetic-trees-top-1000-filtered.json.gz \
#     --visualize-dir results/viz/top_1000/

python scripts/analyze-skeletons.py \
    --skeleton-file results/viz/skeletons.pkl \
    --input-file data/pre-process/syntrees/synthetic-trees-filtered.json.gz \
    --visualize-dir results/viz/

# for split in {'train','valid','test'}; 
# do
#     python scripts/analyze-skeletons.py \
#         --skeleton-file results/viz/skeletons-${split}.pkl \
#         --skeleton-canonical-file results/viz/skeletons.pkl \
#         --input-file data/pre-process/syntrees/synthetic-trees-filtered-${split}.json.gz \
#         --visualize-dir results/viz/
# done
