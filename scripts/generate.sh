export BUILDING_BLOCKS_FILE=data/assets/building-blocks/enamine_us_matched.csv ; 
export RXN_TEMPLATE_FILE=data/assets/reaction-templates/hb.txt ; 
export RXN_COLLECTION_FILE=data/assets/reaction-templates/reactions_hb.json.gz ; 
export EMBEDDINGS_KNN_FILE=data/assets/building-blocks/enamine_us_emb_fp_256.npy ; 
python scripts/03-generate-syntrees.py \
    --building-blocks-file $BUILDING_BLOCKS_FILE \
    --rxn-templates-file $RXN_TEMPLATE_FILE \
    --output-file "data/pre-process/syntrees/synthetic-trees-top-1000.json.gz" \
    --top-bbs-file "results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt" \
    --number-syntrees "600000" \
    --log_file determ_test.debug \
    --ncpu 1
# python scripts/04-filter-syntrees.py \
#     --input-file "data/pre-process/syntrees/synthetic-trees-top-1000.json.gz" \
#     --output-file "data/pre-process/syntrees/synthetic-trees-top-1000-filtered.json.gz" \
#     --verbose
# python scripts/05-split-syntrees.py \
#     --input-file "data/pre-process/syntrees/synthetic-trees-top-1000-filtered.json.gz" \
#     --output-dir "data/pre-process/syntrees/top_1000/" \
#     --verbose
# python scripts/analyze-skeletons.py \
#     --skeleton-file results/viz/top_1000/skeletons-top-1000.pkl \
#     --input-file data/pre-process/syntrees/synthetic-trees-top-1000-filtered.json.gz \
#     --visualize-dir results/viz/top_1000/
# for split in {'train','valid','test'}; do
#     python scripts/analyze-skeletons.py \
#         --skeleton-file results/viz/top_1000/skeletons-top-1000-${split}.pkl \
#         --input-file data/pre-process/syntrees/top_1000/synthetic-trees-filtered-${split}.json.gz \
#         --visualize-dir results/viz/top_1000/ \
#         --skeleton-canonical-file results/viz/top_1000/skeletons-top-1000.pkl
# done
