#!/bin/bash

# delete previous results
rm -rf ./compile
mkdir -p ./compile

FROZEN_PB_PATH="${PWD}/quantize_results/deploy_model.pb"
OUTPUT_DIR="${PWD}/compile"

echo "#####################################"
echo "COMPILE WITH VAI_C_TENSORFLOW"
echo "#####################################"
vai_c_tensorflow \
       --frozen_pb="${FROZEN_PB_PATH}" \
       --arch=./AXU2CGB_DPU_B1152.json \
       --output_dir="${OUTPUT_DIR}" \
       --net_name=cifar10

echo "#####################################"
echo "COMPILATION COMPLETED"
echo "#####################################"