import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
######################################################
# CIFAR-10 training script - IMPROVED VERSION
# Higher accuracy (target 80-85%) with data augmentation,
# batch norm, dropout, and deeper network.
######################################################
import os
import shutil
import tarfile
import pickle
import numpy as np
import tensorflow as tf

# =====================================================
# Setup directories
# =====================================================
SCRIPT_DIR = os.getcwd()
TRAIN_GRAPH = 'training_graph.pb'
CHKPT_FILE = 'float_model.ckpt'
CHKPT_DIR = os.path.join(SCRIPT_DIR, 'chkpts')
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
CIFAR10_DIR = os.path.join(SCRIPT_DIR, 'cifar10_data')
TAR_FILE = os.path.join(SCRIPT_DIR, 'cifar-10-python.tar.gz')

for d in [CIFAR10_DIR, CHKPT_DIR, TB_LOG_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    print(f"Directory {d} created")

# =====================================================
# Extract and load CIFAR-10 dataset
# =====================================================
def extract_cifar10(tar_path, extract_to):
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"Local tar file not found: {tar_path}\n"
                                "Download from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print("Extraction complete.")

def load_cifar_batch(filepath):
    with open(filepath, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    labels = np.array(labels)
    return data, labels

def load_cifar10(extracted_dir):
    x_train_list, y_train_list = [], []
    for i in range(1, 6):
        batch_path = os.path.join(extracted_dir, 'cifar-10-batches-py', f'data_batch_{i}')
        data, labels = load_cifar_batch(batch_path)
        x_train_list.append(data)
        y_train_list.append(labels)
    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list)
    test_path = os.path.join(extracted_dir, 'cifar-10-batches-py', 'test_batch')
    x_test, y_test = load_cifar_batch(test_path)
    return (x_train, y_train), (x_test, y_test)

extract_cifar10(TAR_FILE, CIFAR10_DIR)
(x_train, y_train), (x_test, y_test) = load_cifar10(CIFAR10_DIR)

print(f"Training data shape: {x_train.shape}, labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, labels shape: {y_test.shape}")

# One‑hot encode
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

# Validation split (5000 images)
x_valid = x_train[45000:]
y_valid = y_train[45000:]
x_train = x_train[:45000]
y_train = y_train[:45000]

# =====================================================
# Hyperparameters
# =====================================================
BATCHSIZE = 128
EPOCHS = 150
INIT_LR = 0.001
DECAY_STEPS = 10000
DECAY_RATE = 0.9

# =====================================================
# Data augmentation pipeline (tf.data)
# =====================================================
def augment_image(image, label):
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    # Pad 4 pixels, then random crop back to 32x32
    image = tf.image.resize_image_with_crop_or_pad(image, 32+4, 32+4)
    image = tf.random_crop(image, [32, 32, 3])
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Ensure values stay in [0,1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(50000).repeat(EPOCHS)
train_dataset = train_dataset.map(augment_image, num_parallel_calls=4)
train_dataset = train_dataset.batch(BATCHSIZE, drop_remainder=True)
train_dataset = train_dataset.prefetch(1)

# Validation dataset (no augmentation)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_dataset = valid_dataset.batch(BATCHSIZE)
valid_dataset = valid_dataset.prefetch(1)

# Test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCHSIZE)
test_dataset = test_dataset.prefetch(1)

# =====================================================
# Build the CNN model (deeper, batch norm, dropout)
# =====================================================
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images_in')
y = tf.placeholder(tf.float32, [None, 10], name='labels_in')
training = tf.placeholder(tf.bool, name='training')   # for batch norm and dropout

def conv_bn_relu(inputs, filters, kernel_size=3, strides=1, padding='same'):
    net = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding, use_bias=False)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    return net

def cnn(x):
    # Block 1
    net = conv_bn_relu(x, 32)
    net = conv_bn_relu(net, 32)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.dropout(net, rate=0.2, training=training)

    # Block 2
    net = conv_bn_relu(net, 64)
    net = conv_bn_relu(net, 64)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.dropout(net, rate=0.3, training=training)

    # Block 3
    net = conv_bn_relu(net, 128)
    net = conv_bn_relu(net, 128)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.dropout(net, rate=0.4, training=training)

    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 256, activation=tf.nn.relu)
    net = tf.layers.dropout(net, rate=0.5, training=training)
    logits = tf.layers.dense(net, 10, activation=None)
    return logits

logits = cnn(x)
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y))

# Learning rate decay
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)

# Optimizer with batch norm update ops
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorBoard summaries
tf.summary.scalar('cross_entropy_loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', learning_rate)
tf.summary.image('input_images', x)
tb_summary = tf.summary.merge_all()

saver = tf.train.Saver()

# =====================================================
# Training session
# =====================================================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(TB_LOG_DIR, sess.graph)

    print('-------------------------------------------------------------')
    print('TRAINING PHASE')
    print('-------------------------------------------------------------')

    # Get iterators
    train_iter = train_dataset.make_initializable_iterator()
    valid_iter = valid_dataset.make_initializable_iterator()
    test_iter = test_dataset.make_initializable_iterator()
    next_train = train_iter.get_next()
    next_valid = valid_iter.get_next()
    next_test = test_iter.get_next()

    sess.run(train_iter.initializer)
    steps_per_epoch = len(x_train) // BATCHSIZE

    for epoch in range(EPOCHS):
        sess.run(train_iter.initializer)   # reinitialize each epoch (because .repeat())
        epoch_loss = 0
        epoch_acc = 0

        for step in range(steps_per_epoch):
            batch_x, batch_y = sess.run(next_train)
            _, loss_val, acc_val, lr_val, summ = sess.run(
                [optimizer, loss, accuracy, learning_rate, tb_summary],
                feed_dict={x: batch_x, y: batch_y, training: True})
            epoch_loss += loss_val
            epoch_acc += acc_val
            writer.add_summary(summ, epoch * steps_per_epoch + step)

            if step % 100 == 0:
                print(f'Epoch {epoch+1}/{EPOCHS} Step {step:4d}  LR {lr_val:.5f}  Loss {loss_val:.4f}  Train Acc {acc_val:.4f}')

        avg_loss = epoch_loss / steps_per_epoch
        avg_train_acc = epoch_acc / steps_per_epoch
        print(f'Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Avg Train Acc: {avg_train_acc:.4f}')

        # Validation every epoch
        sess.run(valid_iter.initializer)
        val_acc_sum = 0
        val_steps = len(x_valid) // BATCHSIZE
        for _ in range(val_steps):
            batch_x, batch_y = sess.run(next_valid)
            acc_val = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, training: False})
            val_acc_sum += acc_val
        val_acc = val_acc_sum / val_steps
        print(f'Validation Accuracy: {val_acc:.4f}')

    print('-------------------------------------------------------------')
    print('FINISHED TRAINING')
    print(f'Run `tensorboard --logdir={TB_LOG_DIR} --port 6006 --host localhost` to see results.')
    writer.flush()
    writer.close()

    # Final test accuracy
    sess.run(test_iter.initializer)
    test_acc_sum = 0
    test_steps = len(x_test) // BATCHSIZE
    for _ in range(test_steps):
        batch_x, batch_y = sess.run(next_test)
        acc_val = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, training: False})
        test_acc_sum += acc_val
    test_acc = test_acc_sum / test_steps
    print(f'Final Test Accuracy: {test_acc:.4f}')

    # Save checkpoint and frozen graph
    save_path = saver.save(sess, os.path.join(CHKPT_DIR, CHKPT_FILE))
    print(f'Saved checkpoint to {os.path.join(CHKPT_DIR, CHKPT_FILE)}')
    tf.train.write_graph(sess.graph_def, CHKPT_DIR, TRAIN_GRAPH, as_text=False)
    print(f'Saved binary graphDef to {os.path.join(CHKPT_DIR, TRAIN_GRAPH)}')