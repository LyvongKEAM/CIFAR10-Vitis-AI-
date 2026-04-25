import os
import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util import convert_variables_to_constants

ckpt_dir = './chkpts'
ckpt_prefix = 'float_model.ckpt'
output_graph = './frozen/frozen_graph.pb'

# Reset default graph
tf.reset_default_graph()

# Define the same CNN but with training=False hardcoded
def conv_bn_relu(inputs, filters, kernel_size=3, strides=1, padding='same', training=False):
    net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, use_bias=False)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    return net

def cnn(x, training=False):
    net = conv_bn_relu(x, 32, training=training)
    net = conv_bn_relu(net, 32, training=training)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.dropout(net, rate=0.2, training=training)

    net = conv_bn_relu(net, 64, training=training)
    net = conv_bn_relu(net, 64, training=training)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.dropout(net, rate=0.3, training=training)

    net = conv_bn_relu(net, 128, training=training)
    net = conv_bn_relu(net, 128, training=training)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.dropout(net, rate=0.4, training=training)

    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=training)
    logits = tf.layers.dense(net, 10, activation=None)
    return logits

# Input placeholder (only one)
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images_in')
logits = cnn(x, training=False)
output = tf.identity(logits, name='dense_1/BiasAdd')

# Load weights from checkpoint
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, os.path.join(ckpt_dir, ckpt_prefix))

# Freeze the graph
frozen_graph = convert_variables_to_constants(sess, sess.graph_def, ['dense_1/BiasAdd'])

# Save
os.makedirs('./frozen', exist_ok=True)
graph_io.write_graph(frozen_graph, './frozen', 'frozen_graph.pb', as_text=False)
print("Frozen inference graph saved (no training placeholder).")