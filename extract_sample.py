import numpy as np
import os
import pickle
import cv2  # or use PIL/Pillow if you prefer

# --- Configuration ---
# Path to the directory containing the extracted 'cifar-10-batches-py' folder
DATA_DIR = './cifar10_data'  # Adjust this path as needed
OUTPUT_DIR_TRAIN = './cifar10_jpg/train'
OUTPUT_DIR_TEST = './cifar10_jpg/test'

# Class names for CIFAR-10 (as per batches.meta)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Create output directories for each class
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(OUTPUT_DIR_TRAIN, class_name), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR_TEST, class_name), exist_ok=True)

# --- Helper Function to Unpickle Data ---
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# --- Convert Training Batches (data_batch_1 to data_batch_5) ---
print("Processing training data...")
for i in range(1, 6):
    batch_path = os.path.join(DATA_DIR, 'cifar-10-batches-py', f'data_batch_{i}')
    batch = unpickle(batch_path)
    data = batch[b'data']
    labels = batch[b'labels']
    
    for j in range(len(data)):
        # Reshape and transpose to HWC format
        img = np.reshape(data[j], (3, 32, 32)).transpose(1, 2, 0)
        # OpenCV uses BGR, so convert if needed. For PIL/Pillow, use RGB directly.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        label_name = CLASS_NAMES[labels[j]]
        filename = os.path.join(OUTPUT_DIR_TRAIN, label_name, f'image_{i}_{j}.jpg')
        cv2.imwrite(filename, img)
    print(f"  Finished batch {i}")

# --- Convert Test Batch (test_batch) ---
print("\nProcessing test data...")
test_batch_path = os.path.join(DATA_DIR, 'cifar-10-batches-py', 'test_batch')
test_batch = unpickle(test_batch_path)
test_data = test_batch[b'data']
test_labels = test_batch[b'labels']

for j in range(len(test_data)):
    img = np.reshape(test_data[j], (3, 32, 32)).transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    label_name = CLASS_NAMES[test_labels[j]]
    filename = os.path.join(OUTPUT_DIR_TEST, label_name, f'test_{j}.jpg')
    cv2.imwrite(filename, img)

print("\nDone! Images saved to", OUTPUT_DIR_TRAIN, "and", OUTPUT_DIR_TEST)