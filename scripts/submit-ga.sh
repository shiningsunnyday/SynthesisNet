# cpus=15
# for max_num_rxns in {4,5,6}; 
# do
#     for obj in {'qed','logp','jnk','gsk','drd2'};
#     # for obj in {'qed',};
#     do  
#         for edits in {0,3};
#         # for edits in {0,};
#         do
#             jbsub -proj synthesisnet \
#                 -queue x86_24h \
#                 -name ga.max_num_rxns=${max_num_rxns}_obj=${obj}.edits=${edits} \
#                 -mem 10g \
#                 -cores ${cpus} sh ./sandbox/ga_ours.sh ${obj} ${edits} ${max_num_rxns}
#         done
#     done
# done

# cpus=50
# for seed in {10,};
# do
#     for obj in {'jnk',};
#     do  
#         for strategy in {'edits','flips','topk'};
#         do
#         jbsub -proj synthesisnet \
#             -queue x86_24h \
#             -name ga-analog.obj=${obj}.strategy=${strategy}.new \
#             -mem 20g \
#             -cores ${cpus} sh ./sandbox/ga_ablate.sh "${obj}" "${strategy}" "${seed}"
#         done
#     done
# done

cpus=50
for seed in {10,};
do
    # ,'Celecoxib_Rediscovery','Troglitazone_Rediscovery','Thiothixene_Rediscovery','Aripiprazole_Similarity','Albuterol_Similarity','Mestranol_Similarity','Isomers'
    # for obj in {'Median_1','Median_2','Osimertinib_MPO','Fexofenadine_MPO','Ranolazine_MPO','Perindopril_MPO','Amlodipine_MPO','Sitagliptin_MPO','Zaleplon_MPO'};
    for obj in {'jnk',};
    # for obj in {'jnk',}
    do  
        for strategy in {'flips',};
        do
        jbsub -proj synthesisnet \
            -queue x86_24h \
            -name synnet.obj=${obj}.strategy=${strategy} \
            -mem 200g \
            -cores ${cpus} sh ./sandbox/ga_ablate_synnet.sh "${obj}" "${strategy}" "${seed}"
        done
    done
done
