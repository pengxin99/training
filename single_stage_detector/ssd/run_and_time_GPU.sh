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

DGXSYSTEM=${DGXSYSTEM:-"DGX1_32"}
if [[ -f config_${DGXSYSTEM}.sh ]]; then
  source config_${DGXSYSTEM}.sh
else
  source config_DGX1_32.sh
  echo "Unknown system, assuming DGX1_32"
fi

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
SLURM_JOB_ID=${SLURM_JOB_ID:-$RANDOM}
echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x
MODE=inference
THRESHOLD=0.23
PERF_PRERUN_WARMUP=5
NUMEPOCHS=${NUMEPOCHS:-70}
LR=${LR:-"2.5e-3"}
TOTLE_ITERATIONS=${TOTLE_ITERATIONS:-0}
BATCH_SIZE=32
LR=${LR:-"2.5e-3"}
ARCH="ssd300"
DUMMY=0 # (0)real data,(1)dummy data

echo "running benchmark"

export DATASET_DIR="/data/coco2017"
export TORCH_MODEL_ZOO="/data/torchvision"

python -m bind_launch --nsockets_per_node ${DGXNSOCKET} \
                      --ncores_per_socket ${DGXSOCKETCORES} \
                      --nproc_per_node $SLURM_NTASKS_PER_NODE $MULTI_NODE \
 performance_test.py \
   -a $ARCH \
   -b $BATCH_SIZE \
  --epochs "${NUMEPOCHS}" \
  --warmup-factor 0 \
  --lr "${LR}" \
  --no-save \
  --threshold=$THRESHOLD \
  --data ${DATASET_DIR} \
  --totle-iteration "${TOTLE_ITERATIONS}" \
  -m $MODE \
  --perf-prerun-warmup $PERF_PRERUN_WARMUP \
  ${EXTRA_PARAMS[@]} ; ret_code=$?

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
