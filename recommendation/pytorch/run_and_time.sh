#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=1.0
#BASEDIR='/data/cache'
#DATASET=${DATASET:-ml-20m}
BASEDIR='/home/pengxiny/pytorch_work/training/recommendation/data_set'
DATASET=${DATASET:-ml-20m}

# Get command line seed
usage()
{
    echo "usage: bash ./run_and_time.sh [[[-s seed ] [-inf inference] [-model pretrained_model]] | [-h]]"
}
SEED=0
IS_INF=0
PRETRAINED_MODEL=None

while [ $# -gt 0 ]; do
  case "$1" in
    --seed | -s)
      shift
      SEED=$1
      ;;
    --inference | -inf)
      shift
      IS_INF=$1
      ;;
    --pretrained_model | -model)
      shift
      PRETRAINED_MODEL=$1
      ;;
    *)
      usage
      exit 1
  esac
  shift
done

echo "seed : ${SEED}"
echo "IS_INF : ${IS_INF}"
echo "PRETRAINED_MODEL : ${PRETRAINED_MODEL}"

# Get the multipliers for expanding the dataset
#USER_MUL=${USER_MUL:-16}
#ITEM_MUL=${ITEM_MUL:-32}
USER_MUL=${USER_MUL:-1}
ITEM_MUL=${ITEM_MUL:-1}

DATASET_DIR=${BASEDIR}/${DATASET}x${USER_MUL}x${ITEM_MUL}

export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=56

if [ -d ${DATASET_DIR} ]
then
    # start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"

    numactl --physcpubind=0-55 --membind=0 python -u ncf.py ${DATASET_DIR} \
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
        --inf ${IS_INF} \
        --pretrained_model ${PRETRAINED_MODEL}
	# end timing
	end=$(date +%s)
	end_fmt=$(date +%Y-%m-%d\ %r)
	echo "ENDING TIMING RUN AT $end_fmt"


	# report result
	result=$(( $end - $start ))
	result_name="recommendation"


	echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
else
	echo "Directory ${DATASET_DIR} does not exist"
fi





