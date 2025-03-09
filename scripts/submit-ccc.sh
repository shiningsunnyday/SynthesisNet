cpus=5
for ((i =1; i <= 50; i++));
do
    jbsub -proj synthesisnet \
        -queue x86_24h \
        -name listener.${i} \
        -mem 10g \
        -cores ${cpus} sh ./scripts/reconstruct-listener-ccc.sh $i
    # jbsub -proj synthesisnet \
    #     -name mcmc.listener.${i} \
    #     -mem 10g \
    #     -cores ${cpus} sh ./scripts/mcmc-analog-listener-ccc.sh $i
done
