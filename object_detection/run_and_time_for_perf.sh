#!/bin/bash

# Runs benchmark and reports time to convergence

pushd pytorch

MODE=inference
CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
if [ $MODE = training ]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    NUM_THREADS=$TOTAL_CORES
else
    TOTAL_CORES=$CORES
    NUM_THREADS=$TOTAL_CORES
fi

for i in $(seq 0 $(($TOTAL_CORES / $NUM_THREADS - 1)))
do
echo $i "instance"
startid=$(($i*$NUM_THREADS))
endid=$(($i*$NUM_THREADS+$NUM_THREADS-1))
export OMP_SCHEDULE=STATIC OMP_NUM_THREADS=$NUM_THREADS OMP_DISPLAY_ENV=TRUE OMP_PROC_BIND=TRUE GOMP_CPU_AFFINITY="$startid-$endid"  
export OMP_NUM_THREADS=$NUM_THREADS  KMP_AFFINITY=proclist=[$startid-$endid],granularity=fine,explicit

if [ $MODE = training ]; then
    time python tools/train_mlperf.py --use-mkldnn --warmup 5 --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" \
       SOLVER.BASE_LR 0.0025 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 MODEL.DEVICE "cpu" DATALOADER.NUM_WORKERS 1 &
else
    time python tools/test_net.py --warmup 2 --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_for_cpu.yaml" \
       TEST.IMS_PER_BATCH 16 MODEL.DEVICE "cpu" DATALOADER.NUM_WORKERS 1 &
fi

done
set +x

# sleep 3
# if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# # end timing
# end=$(date +%s)
# end_fmt=$(date +%Y-%m-%d\ %r)
# echo "ENDING TIMING RUN AT $end_fmt"

# # report result
# result=$(( $end - $start ))
# result_name="OBJECT_DETECTION"

# echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
       
popd
