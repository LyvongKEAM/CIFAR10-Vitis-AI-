#!/bin/bash
echo "#####################################"
echo "QUANTIZE CIFAR-10"
echo "#####################################"

vai_q_tensorflow quantize \
  --input_frozen_graph ./frozen/frozen_graph.pb \
  --input_nodes images_in \
  --input_shapes ?,32,32,3 \
  --output_nodes dense_1/BiasAdd \
  --method 1 \
  --input_fn graph_input_fn.calib_input \
  --gpu 0 \
  --calib_iter 400

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"