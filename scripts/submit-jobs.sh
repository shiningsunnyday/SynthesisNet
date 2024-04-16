#!/bin/bash

# Expand (depth=1, batch=-1), log.txt
# jbsub \
#     -proj syntreenet \
#     -mem 20g \
#     -cores 40 \
#     python scripts/build-hash-table.py \
#     --ncpu 40 \
#     --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
#     --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log.txt \
#     --step expand \
#     --batch -1 \
#     --keep-prods 2 \
#     --depth 3 \
#     --expand_batch_size 1 \
#     --init_batch_size 1 \
#     --run_batch_size 1 \
#     --mp-min-combinations 10000000000000000000000000000000000

# Init (depth=1, batch=0, ..., K), log-init-$batch.txt
# for batch in {0..90}
#     do
#         jbsub \
#             -proj syntreenet \
#             -name batch.${batch} \
#             -mem 10g \
#             python scripts/build-hash-table.py \
#             --ncpu 1 \
#             --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
#             --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log-init-${batch}.txt \
#             --step init \
#             --batch $batch \
#             --keep-prods 2 \
#             --depth 3 \
#             --expand_batch_size 1 \
#             --init_batch_size 1 \
#             --run_batch_size 1 \
#             --mp-min-combinations 10000000000000000000000000000000000
#     done

# Init (depth=1, batch=-1), log.txt
# jbsub \
#     -proj syntreenet \
#     -mem 20g \
#     python scripts/build-hash-table.py \
#     --ncpu 1 \
#     --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
#     --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log.txt \
#     --step init \
#     --batch -1 \
#     --keep-prods 2 \
#     --depth 3 \
#     --expand_batch_size 1 \
#     --init_batch_size 1 \
#     --run_batch_size 1 \
#     --mp-min-combinations 10000000000000000000000000000000000

# Run (depth=1, batch=0, ..., K), log-run-$batch.txt
for batch in {0..88}
    do
        file="/u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log-run-${batch}.txt";
        if [[ -f "/dccstor/graph-design/program_cache_keep-prods=2/1_run_${batch}.pkl" ]]; then
            exist=true;
        else
            exist=false;
        fi
        # while read -r line; do
        #     # Do whatever you want to do with each line here
        #     if [[ $line == *"done dumping batch programs"* ]]; then        
        #         # echo $line;
        #         exist=true;
        #     fi
        # done < "$file"
        # echo ${exist}
        if [ $exist == false ]; then
            echo $batch
            # jbsub \
            #     -proj syntreenet \
            #     -queue x86_24h \
            #     -name batch.${batch}.hard \
            #     -mem 1000g \
            #     -cores 20 \
            #     python scripts/build-hash-table.py \
            #     --ncpu 20 \
            #     --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
            #     --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log-run-${batch}.txt \
            #     --step run \
            #     --batch $batch \
            #     --keep-prods 2 \
            #     --depth 3 \
            #     --expand_batch_size 1 \
            #     --init_batch_size 1 \
            #     --run_batch_size 1 \
            #     --mp-min-combinations 10000000
        fi
    done

# Run (depth=1, batch=-1), log.txt
# jbsub \
#     -proj syntreenet \
#     -queue x86_24h \
#     -name batch.-1 \
#     -mem 50g \
#     python scripts/build-hash-table.py \
#     --ncpu 1 \
#     --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
#     --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log.txt \
#     --step run \
#     --batch -1 \
#     --keep-prods 2 \
#     --depth 3 \
#     --expand_batch_size 1 \
#     --init_batch_size 1 \
#     --run_batch_size 1 \
#     --mp-min-combinations 10000000
# Prepare expand (pargs per batch), log.txt

# jbsub \
#     -proj syntreenet \
#     -mem 100g \
#     -cores 1 \
#     python scripts/build-hash-table.py \
#     --ncpu 1 \
#     --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
#     --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log.txt \
#     --step expand \
#     --batch -1 \
#     --d 2 \
#     --keep-prods 2 \
#     --depth 3 \
#     --expand_batch_size 1 \
#     --init_batch_size 1 \
#     --run_batch_size 1 \
#     --mp-min-combinations 10000000000000000000000000000000000

# Expand (depth=2, batch=0, ..., K), log-expand-$batch.txt
# for batch in {0..1504};
# do
#     jbsub \
#         -proj syntreenet \
#         -name batch.${batch} \
#         -mem 100g \
#         -cores 1 \
#         python scripts/build-hash-table.py \
#         --ncpu 1 \
#         --cache-dir /dccstor/graph-design/program_cache_keep-prods=2/ \
#         --log_file /u/msun415/SynTreeNet/results/viz/program_cache_keep-prods=2/log-expand-2-${batch}.txt \
#         --step expand \
#         --batch ${batch} \
#         --d 2 \
#         --keep-prods 2 \
#         --depth 3 \
#         --expand_batch_size 10 \
#         --init_batch_size 1 \
#         --run_batch_size 1 \
#         --mp-min-combinations 10000000000000000000000000000000000
# done;
# Expand (depth=2, batch=-1), log.txt
# Init (depth=2, batch=0, ..., K), log-init-$batch.txt
# Init (depth=2, batch=-1), log.txt
# Run (depth=2, batch=0, ..., K), log-run-$batch.txt
# Run (depth=2, batch=-1), log.txt
