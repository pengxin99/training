#!/bin/bash
# DeepSpeech2 multi_instances training

NUM_THREAD=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
NUM_NUMA=`lscpu | grep 'NUMA node(s):' | awk '{print $3}'`

if [ ! -d "training_throughput_logs" ];then
    mkdir training_throughput_logs
fi


# model configurations
RANDOM_SEED=1
TARGET_ACC=23
BATCH_SIZE=8

time1=$(date "+%Y%m%d%H%M%S")_$BATCH_SIZE
mkdir training_throughput_logs/$time1

for i in $(seq 0 $(($NUM_NUMA-1)))
do
    startid=$(($i*$NUM_THREAD))
    endid=$(($startid+$NUM_THREAD-1))
    KMP_BLOCKTIME=1 KMP_HW_SUBSET=1t KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=$NUM_THREAD numactl -C$startid-$endid -m$i \
    python -u train.py --model_path models/deepspeech_t$RANDOM_SEED.pth.tar \
                       --seed $RANDOM_SEED \
                       --acc $TARGET_ACC \
                       |& tee training_throughput_logs/$time1/instance_$i.log  &
done
