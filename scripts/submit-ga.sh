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
for max_num_rxns in {4,}; 
do
    for obj in {'jnk','gsk','qed','drd2'};
    # for obj in {'qed',};
    do  
        # for edits in {0,1,2,3};
        # # for edits in {0,};
        # do
        #     jbsub -proj syntreenet \
        #         -queue x86_24h \
        #         -name ga.max_num_rxns=${max_num_rxns}_obj=${obj}.edits=${edits} \
        #         -mem 10g \
        #         -cores ${cpus} sh ./sandbox/ga_ablate.sh ${obj} ${edits}
        # done
        jbsub -proj syntreenet \
            -queue x86_24h \
            -name ga-analog.obj=${obj} \
            -mem 10g \
            -cores ${cpus} sh ./sandbox/ga_ablate.sh ${obj}
        
    done
done