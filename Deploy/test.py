import os
import cv2
import numpy as np
from dnndk import n2cube

# ========== CONFIGURATION ==========
KERNEL_NAME = "cifar10"
INPUT_NODE = "conv2d_Conv2D"
OUTPUT_NODE = "dense_1_MatMul"
LABELS_FILE = "./cifar10_lables.txt"
TEST_DIR = "./dataset"
TEST_IMAGE = "test_1.jpg"
# ===================================

def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load {img_path}")

    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def main():
    labels = load_labels(LABELS_FILE)
    print(f"Classes: {labels}")

    img_path = os.path.join(TEST_DIR, TEST_IMAGE)

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    # Initialize DPU
    n2cube.dpuOpen()
    kernel = n2cube.dpuLoadKernel(KERNEL_NAME)
    if not kernel:
        print("ERROR: Kernel load failed")
        n2cube.dpuClose()
        return

    task = n2cube.dpuCreateTask(kernel, 0)

    try:
        input_data = preprocess(img_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"\nProcessing {TEST_IMAGE}")

    # Set input
    input_size = n2cube.dpuGetInputTensorSize(task, INPUT_NODE)
    n2cube.dpuSetInputTensorInHWCFP32(task, INPUT_NODE, input_data, input_size)

    # Run DPU
    n2cube.dpuRunTask(task)

    # Get output
    output_size = n2cube.dpuGetOutputTensorSize(task, OUTPUT_NODE)
    output_data = n2cube.dpuGetOutputTensorInHWCFP32(task, OUTPUT_NODE, output_size)

    pred_idx = np.argmax(output_data)

    print(f"Predicted: {labels[pred_idx]} (class {pred_idx})")

    # Cleanup
    n2cube.dpuDestroyTask(task)
    n2cube.dpuDestroyKernel(kernel)
    n2cube.dpuClose()

if __name__ == "__main__":
    main()