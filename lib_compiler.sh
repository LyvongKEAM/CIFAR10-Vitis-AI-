#!/bin/bash

# Assume the kernel name is always "cifar10" (lowercase)
NET_NAME="cifar10"
ELF_FILE="dpu_${NET_NAME}.elf"          # dpu_cifar10.elf
KERNEL_PATH="./compile"

# Check if the .elf file exists
if [ ! -f "${KERNEL_PATH}/${ELF_FILE}" ]; then
    echo "ERROR: ${ELF_FILE} not found in ${KERNEL_PATH}"
    echo "Available .elf files:"
    ls -1 ${KERNEL_PATH}/*.elf 2>/dev/null || echo "  None"
    exit 1
fi

cd "${KERNEL_PATH}" || exit 1

# Create shared object with the exact name libdpumodelcifar10.so
aarch64-xilinx-linux-gcc --sysroot=/opt/vitis_ai/petalinux_sdk/sysroots/aarch64-xilinx-linux \
    -fPIC -shared "${ELF_FILE}" -o "libdpumodel${NET_NAME}.so"

echo "Shared library libdpumodel${NET_NAME}.so created in ${KERNEL_PATH}"