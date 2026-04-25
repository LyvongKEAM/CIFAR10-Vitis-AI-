import numpy as np
import pickle
import os

batch_dir = './cifar10_data/cifar-10-batches-py'
output_file = './cifar10_calib.npz'

with open(os.path.join(batch_dir, 'data_batch_1'), 'rb') as f:
    batch = pickle.load(f, encoding='bytes')

# Shape: (10000, 3072) -> (10000, 3, 32, 32) -> (10000, 32, 32, 3)
images = batch[b'data']
images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0

calib_images = images[:1000]

np.savez(output_file, images=calib_images)
print(f"Saved {calib_images.shape} images to {output_file}")