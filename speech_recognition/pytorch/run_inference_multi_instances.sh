#!/bin/bash
# DeepSpeech2 multi_instances inference

NUM_THREAD=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
NUM_NUMA=`lscpu | grep 'NUMA node(s):' | awk '{print $3}'`
NUM_CORES=$NUM_THREAD

if [ ! -d "inference_realtime_logs" ];then
    mkdir inference_realtime_logs
fi
if [ ! -d "inference_throughput_logs" ];then
    mkdir inference_throughput_logs
fi

if [ ! $1 ];then
    BATCH_SIZE=8
else
    BATCH_SIZE=$1
fi

save_log='inference_throughput_logs'
if [[ "$1" == "--single" ]]; then
  echo "### using single batch size"
  BATCH_SIZE=1
  NUM_THREAD=4
  save_log='inference_realtime_logs'
  shift
fi 

time1=$(date "+%Y%m%d%H%M%S")_$BATCH_SIZE
mkdir $save_log/$time1

RANDOM_SEED=1
MODEL_PATH='models/deepspeech_10.pth'


INT_PER_NODE=$(($NUM_CORES / $NUM_THREAD - 1))
for i in $(seq 0 $(($NUM_NUMA-1)))
do
for j in $(seq 0 $INT_PER_NODE)
do
    startid=$(($i*$NUM_CORES+$j*$NUM_THREAD))
    endid=$(($i*$NUM_CORES+$j*$NUM_THREAD+$NUM_THREAD-1))
    OMP_NUM_THREADS=$NUM_THREAD numactl --physcpubind=$startid-$endid --membind=$i \
    python -u inference.py --seed $RANDOM_SEED \
                           --model_path $MODEL_PATH \
                           --batch_size $BATCH_SIZE \
                           2>&1|& tee $save_log/$time1/$startid-$endid.log &

done
done