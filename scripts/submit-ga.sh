# cpus=15
# for max_num_rxns in {4,5,6}; 
# do
#     for obj in {'qed','logp','jnk','gsk','drd2'};
#     # for obj in {'qed',};
#     do  
#         for edits in {0,3};
#         # for edits in {0,};
#         do
#             jbsub -proj syntreenet \
#                 -queue x86_24h \
#                 -name ga.max_num_rxns=${max_num_rxns}_obj=${obj}.edits=${edits} \
#                 -mem 10g \
#                 -cores ${cpus} sh ./sandbox/ga_ours.sh ${obj} ${edits} ${max_num_rxns}
#         done
#     done
# done

cpus=50
for seed in {10,};
do
    for obj in {'jnk',};
    do  
        for strategy in {'edits','flips','topk'};
        do
        jbsub -proj syntreenet \
            -queue x86_24h \
            -name ga-analog.obj=${obj}.strategy=${strategy} \
            -mem 20g \
            -cores ${cpus} sh ./sandbox/ga_ablate.sh "${obj}" "${strategy}" "${seed}"
        done
    done
done
