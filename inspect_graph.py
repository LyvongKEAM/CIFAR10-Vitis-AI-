import tensorflow as tf
import cv2
import numpy as np

FLOAT_PB = "./frozen/frozen_graph.pb"
IMAGE_PATH = "./cifar10_jpg/test/cat/test_0.jpg"  # change if needed

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

with tf.gfile.GFile(FLOAT_PB, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name="")

with tf.Session() as sess:
    input_tensor = sess.graph.get_tensor_by_name("images_in:0")
    output_tensor = sess.graph.get_tensor_by_name("dense_1/BiasAdd:0")
    img = preprocess(IMAGE_PATH)
    print(f"Input range: [{img.min():.3f}, {img.max():.3f}]")
    logits = sess.run(output_tensor, feed_dict={input_tensor: img})
    print(f"Float logits: {logits}")
    probs = tf.nn.softmax(logits).eval()
    pred = np.argmax(probs)
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print(f"Predicted: {classes[pred]}, confidence: {probs[0][pred]:.4f}")