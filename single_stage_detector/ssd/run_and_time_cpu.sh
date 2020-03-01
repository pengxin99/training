#!/bin/bash

# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time_cpu.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
MODE=training
BATCH_SIZE=32
THRESHOLD=0.23
PERF_PRERUN_WARMUP=5
NUMEPOCHS=${NUMEPOCHS:-1}
LR=${LR:-"2.5e-3"}
TOTLE_ITERATIONS=${TOTLE_ITERATIONS:-10}

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
if [ $MODE = training ]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    NUM_THREADS=$TOTAL_CORES
else
    TOTAL_CORES=$CORES
    NUM_THREADS=$TOTAL_CORES
fi

echo "running benchmark"
for i in $(seq 0 $(($TOTAL_CORES / $NUM_THREADS - 1)))
do
echo $i "instance"
startid=$(($i*$NUM_THREADS))
endid=$(($i*$NUM_THREADS+$NUM_THREADS-1))
export OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$NUM_THREADS OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="$startid-$endid"  
export OMP_NUM_THREADS=$NUM_THREADS  KMP_AFFINITY=proclist=[$startid-$endid],granularity=fine,explicit
export DATASET_DIR="/lustre/dataset/COCO2017"
# export TORCH_MODEL_ZOO="/data/torchvision"

python performance_test.py \
  --epochs "${NUMEPOCHS}" \
  --warmup-factor 0 \
  --lr "${LR}" \
  --no-save \
  --threshold=$THRESHOLD \
  --data ${DATASET_DIR} \
  --no-cuda \
  --use-mkldnn \
  -b $BATCH_SIZE \
  --totle-iteration "${TOTLE_ITERATIONS}" \
  -m $MODE \
  --perf-prerun-warmup $PERF_PRERUN_WARMUP \
  ${EXTRA_PARAMS[@]}
ret_code=$?
done
set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
