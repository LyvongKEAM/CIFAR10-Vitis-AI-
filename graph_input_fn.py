# import numpy as np
# import pickle
# import os

# _calib_images = None
# _batch_size = 50

# def _load_calib_images():
#     global _calib_images
#     if _calib_images is None:
#         batch_dir = './cifar10_data/cifar-10-batches-py'
#         batch_file = os.path.join(batch_dir, 'data_batch_1')
#         with open(batch_file, 'rb') as f:
#             batch = pickle.load(f, encoding='bytes')
#         images = batch[b'data']
#         images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
#         _calib_images = images[:20000]
#     return _calib_images

# def calib_input(iter):
#     images = _load_calib_images()
#     start = iter * _batch_size
#     end = start + _batch_size
#     batch = images[start:end]
#     return {'images_in': batch}




import numpy as np
import os
import cv2

_calib_images = []
_batch_size = 50
image_dir = "./cifar10_jpg/train"   # use train folder

def _load_images():
    global _calib_images

    if not _calib_images:
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(root, f)

                    img = cv2.imread(path)
                    if img is None:
                        continue

                    img = cv2.resize(img, (32, 32))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # MUST match training
                    img = img.astype(np.float32) / 255.0

                    _calib_images.append(img)

        print("Loaded images:", len(_calib_images))

    return _calib_images


def calib_input(iter):
    images = _load_images()

    if len(images) == 0:
        raise ValueError("No calibration images found!")

    idx = np.random.choice(len(images), _batch_size)
    batch = np.array([images[i] for i in idx])

    return {"images_in": batch}