#!/bin/bash

set -x

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

model_dir="/mnt/cephfs/hjh/train_record/vc/freevc/train_vctk/train_output_freevc_s"

for((i=0;i<=100;i+=1))
do
  CUDA_VISIBLE_DEVICES=6,7 \
  python train.py -c `pwd`/configs/freevc-s.json -m ${model_dir} ||true
done
