#!/bin/bash

NUM_THREAD=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
NUM_NUMA=`lscpu | grep 'NUMA node(s):' | awk '{print $3}'`
NUM_CORES=$NUM_THREAD

if [ ! -d "inference_realtime_logs" ];then
    mkdir inference_realtime_logs
fi
if [ ! -d "inference_throughput_logs" ];then
    mkdir inference_throughput_logs
fi



# Get command line seed
#usage()
#{
#    echo "usage: bash ./run_and_time.sh [[[-s seed ] [-inf inference] [-model pretrained_model]] | [-h]]"
#}

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
PRETRAINED_MODEL="./model/5_epoch_model.pkl"
BATCH_SIZE=2**20
save_log='inference_throughput_logs'

while [ $# -gt 0 ]; do
  case "$1" in
    --single | -single)
      shift
      echo "### using single batch size"
      BATCH_SIZE=1
      NUM_THREAD=4
      save_log='inference_realtime_logs'
      ;;
    --seed | -s)
      shift
      SEED=$1
      ;;
    --pretrained_model | -model)
      shift
      PRETRAINED_MODEL=$1
      ;;
    *)
#    usage
#      exit 1
  esac
  shift
done

time1=$(date "+%Y%m%d%H%M%S")_$BATCH_SIZE
mkdir $save_log/$time1

echo "seed : $"
# echo "IS_INF : ${IS_INF}"
echo "PRETRAINED_MODEL : ${PRETRAINED_MODEL}"

INT_PER_NODE=$(($NUM_CORES / $NUM_THREAD - 1))
for i in $(seq 0 $(($NUM_NUMA-1)))
do
for j in $(seq 0 $INT_PER_NODE)
do
    startid=$(($i*$NUM_CORES+$j*$NUM_THREAD))
    endid=$(($i*$NUM_CORES+$j*$NUM_THREAD+$NUM_THREAD-1))
    OMP_NUM_THREADS=$NUM_THREAD numactl --physcpubind=$startid-$endid --membind=$i \
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
        --inf 1 \
        --pretrained_model ${PRETRAINED_MODEL} \
        2>&1|& tee $save_log/$time1/$startid-$endid.log &

done
done

