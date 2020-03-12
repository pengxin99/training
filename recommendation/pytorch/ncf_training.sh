#!/bin/bash

NUM_THREAD=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
NUM_NUMA=`lscpu | grep 'NUMA node(s):' | awk '{print $3}'`
NUM_CORES=$NUM_THREAD


if [ ! -d "training_throughput_logs" ];then
    mkdir training_throughput_logs
fi

# model configuration
THRESHOLD=1.0
BASEDIR='../data_set'
DATASET=${DATASET:-ml-20m}

# Get the multipliers for expanding the dataset
USER_MUL=${USER_MUL:-1}
ITEM_MUL=${ITEM_MUL:-1}
DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}

SEED=0
IS_INF=0
BATCH_SIZE=65536
save_log='training_throughput_logs'

time1=$(date "+%Y%m%d%H%M%S")_$BATCH_SIZE
mkdir $save_log/$time1


INT_PER_NODE=$(($NUM_CORES / $NUM_THREAD - 1))

for i in $(seq 0 $(($NUM_NUMA-1)))
do
    startid=$(($i*$NUM_THREAD))
    endid=$(($startid+$NUM_THREAD-1))
    KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$NUM_THREAD numactl --physcpubind=$startid-$endid --membind=$i \
    python -u ncf.py ${DATASET_DIR} \
        -l 0.0002 \
        -b 65536 \
        --layers 256 256 128 64 \
        -f 64 \
        -e 10 \
        --seed $SEED \
        --threshold $THRESHOLD \
        --user_scaling ${USER_MUL} \
        --item_scaling ${ITEM_MUL} \
        --cpu_dataloader \
        --random_negatives \
        --inf 0 \
        2>&1|& tee $save_log/$time1/$startid-$endid.log &

done

