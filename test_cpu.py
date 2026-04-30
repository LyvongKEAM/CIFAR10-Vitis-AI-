# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import os

# # =====================================================
# # CONFIG
# # =====================================================
# CHKPT_DIR = 'chkpts'   # same folder from training
# IMAGE_PATH = 'test_22.jpg'  # your test image

# # CIFAR-10 class labels
# class_names = [
#     'airplane','automobile','bird','cat','deer',
#     'dog','frog','horse','ship','truck'
# ]

# # =====================================================
# # IMAGE PREPROCESS FUNCTION
# # =====================================================
# def load_image(image_path):
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     img = Image.open(image_path).convert('RGB')
#     img = img.resize((32, 32))  # CIFAR-10 size
#     img = np.array(img).astype(np.float32) / 255.0
#     img = np.expand_dims(img, axis=0)  # (1,32,32,3)
#     return img

# # =====================================================
# # LOAD MODEL + RUN INFERENCE
# # =====================================================
# tf.reset_default_graph()

# with tf.Session() as sess:
#     print("Loading model...")

#     # Load checkpoint
#     saver = tf.train.import_meta_graph(os.path.join(CHKPT_DIR, 'float_model.ckpt.meta'))
#     saver.restore(sess, tf.train.latest_checkpoint(CHKPT_DIR))

#     print("Model loaded successfully!")

#     graph = tf.get_default_graph()

#     # =====================================================
#     # GET REQUIRED TENSORS
#     # =====================================================
#     x = graph.get_tensor_by_name('images_in:0')
#     training = graph.get_tensor_by_name('training:0')

#     # ⚠️ This name may vary depending on your graph
#     try:
#         logits = graph.get_tensor_by_name('dense_1/BiasAdd:0')
#     except:
#         print("Could not find 'dense_1/BiasAdd:0'. Listing layers...")
#         for op in graph.get_operations():
#             print(op.name)
#         raise ValueError("Please update logits tensor name!")

#     # Softmax for probability
#     probs = tf.nn.softmax(logits)

#     # =====================================================
#     # LOAD IMAGE
#     # =====================================================
#     img = load_image(IMAGE_PATH)

#     # =====================================================
#     # RUN PREDICTION
#     # =====================================================
#     prediction = sess.run(probs, feed_dict={
#         x: img,
#         training: False
#     })

#     pred_class = np.argmax(prediction)
#     confidence = np.max(prediction)

#     # =====================================================
#     # RESULT
#     # =====================================================
#     print("\n========== RESULT ==========")
#     print(f"Predicted Class : {class_names[pred_class]}")
#     print(f"Confidence      : {confidence:.4f}")
#     print("============================")


import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

# =====================================================
# CONFIG
# =====================================================
CHKPT_DIR = 'chkpts'
IMAGE_PATH = 'test_22.jpg'

class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# =====================================================
# IMAGE PREPROCESS FUNCTION
# =====================================================
def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =====================================================
# LOAD MODEL + RUN INFERENCE
# =====================================================
tf.reset_default_graph()

with tf.Session() as sess:
    print("Loading model...")

    saver = tf.train.import_meta_graph(os.path.join(CHKPT_DIR, 'float_model.ckpt.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(CHKPT_DIR))

    print("Model loaded successfully!")

    graph = tf.get_default_graph()

    # =====================================================
    # GET REQUIRED TENSORS
    # =====================================================
    x = graph.get_tensor_by_name('images_in:0')
    training = graph.get_tensor_by_name('training:0')

    try:
        logits = graph.get_tensor_by_name('dense_1/BiasAdd:0')
    except:
        print("Could not find 'dense_1/BiasAdd:0'. Listing layers...")
        for op in graph.get_operations():
            print(op.name)
        raise ValueError("Please update logits tensor name!")

    probs = tf.nn.softmax(logits)

    # =====================================================
    # LOAD IMAGE
    # =====================================================
    img = load_image(IMAGE_PATH)

    # =====================================================
    # WARM-UP (IMPORTANT for fair timing)
    # =====================================================
    for _ in range(5):
        sess.run(probs, feed_dict={x: img, training: False})

    # =====================================================
    # TIMING (MULTIPLE RUNS)
    # =====================================================
    runs = 50
    times = []

    for _ in range(runs):
        start = time.time()

        prediction = sess.run(probs, feed_dict={
            x: img,
            training: False
        })

        end = time.time()
        times.append((end - start) * 1000)  # ms

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # =====================================================
    # RESULT
    # =====================================================
    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print("\n========== RESULT ==========")
    print(f"Predicted Class : {class_names[pred_class]}")
    print(f"Confidence      : {confidence:.4f}")
    print("============================")

    print("\n========== CPU PERFORMANCE ==========")
    print(f"CPU runtime : {avg_time:.2f} ms")
    print("====================================")