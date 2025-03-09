export cpus=10;
dir="/dccstor/graph-design/gnn/";
# for i in {1,2,3,4,7,8,9,10,13,16,18,19,24,35,36,41,75} ; 
for i in {-1,} ; 
do
    jbsub -proj synthesisnet \
        -queue x86_24h \
        -name train.${i}.rxn \
        -mem 50g \
        -cores ${cpus}+1 sh ./scripts/submit_train_job.sh $i ${cpus} 0 ${dir}
    jbsub -proj synthesisnet \
        -queue x86_24h \
        -name train.${i}.bb \
        -mem 50g \
        -cores ${cpus}+1 sh ./scripts/submit_train_job.sh $i ${cpus} 1 ${dir}
done
