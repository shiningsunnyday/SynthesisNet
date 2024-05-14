cpus=10
for ((i =1; i <= 500; i++));
do
    jbsub -proj syntreenet \
        -queue x86_24h \
        -name listener.${i} \
        -mem 10g \
        -cores ${cpus} sh ./scripts/reconstruct-listener-ccc.sh $i
done
