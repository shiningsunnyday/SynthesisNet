export BUILDING_BLOCKS_FILE=data/assets/building-blocks/enamine_us_matched.csv
export RXN_TEMPLATE_FILE=data/assets/reaction-templates/hb.txt
export RXN_COLLECTION_FILE=data/assets/reaction-templates/reactions_hb.json.gz
export EMBEDDINGS_KNN_FILE=data/assets/building-blocks/enamine_us_emb_fp_256.npy
export OMP_NUM_THREADS=1



# python src/synnet/models/rt1.py --data-dir data/featurized_orig/Xy --mol-embedder-file $EMBEDDINGS_KNN_FILE
for obj in {'jnk','qed','gsk','drd2'}; do
    # python scripts/optimize_ga.py \
    #     --ckpt-dir "results/logs/" \
    #     --building-blocks-file $BUILDING_BLOCKS_FILE \
    #     --rxns-collection-file $RXN_COLLECTION_FILE \
    #     --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
    #     --radius 2 \
    #     --nbits 4096 \
    #     --num_population 128 \
    #     --num_offspring 512 \
    #     --num_gen 200 \
    #     --objective ${obj} \
    #     --ncpu 50 \
    #     --input-file indvs-${obj}.json \
    #     --ckpt-versions 4 24 2 3 \
    #     --top-bbs-file "results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt"
    python scripts/optimize_ga.py \
        --building-blocks-file $BUILDING_BLOCKS_FILE \
        --rxns-collection-file $RXN_COLLECTION_FILE \
        --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
        --radius 2 \
        --nbits 4096 \
        --num_population 128 \
        --num_offspring 512 \
        --num_gen 200 \
        --objective ${obj} \
        --ncpu 50 \
        --ckpt-dir results/baseline_ckpt/ \
        --input-file data/assets/molecules/zinc.csv \
        # --input-file indvs-${obj}.json \                
        # --top-bbs-file "results/viz/programs/program_cache-bb=1000-prods=2/bblocks-top-1000.txt"    
done

# python scripts/20-predict-targets.py \
#     --building-blocks-file $BUILDING_BLOCKS_FILE \
#     --rxns-collection-file $RXN_COLLECTION_FILE \
#     --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
#     --data test \
#     --ckpt-dir "results/logs/" \
#     --output-dir "results/baseline/" \
#     --ckpt-versions 5 25 3 4 \
#     --ncpu 50

    # --data data/assets/molecules/chembl_34_chemreps.tsv \

# python scripts/20-predict-targets.py \
#     --building-blocks-file $BUILDING_BLOCKS_FILE \
#     --rxns-collection-file $RXN_COLLECTION_FILE \
#     --embeddings-knn-file $EMBEDDINGS_KNN_FILE \
#     --data data/assets/molecules/chembl_34_chemreps.tsv \
#     --num 1000 \
#     --ckpt-dir "results/logs/" \
#     --output-dir "results/baseline/" \
#     --ckpt-versions 5 25 3 4 \
#     --ncpu 50

# for depth in {3,4,5,6}; do
#     python scripts/23-evaluate-predictions.py \
#         # --input-file "output_reconstruct_top_k=3_max_num_rxns=${depth}_max_rxns=-1.csv"
# done;