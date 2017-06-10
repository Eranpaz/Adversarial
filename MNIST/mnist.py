from setting import cfg
import tensorflow as tf
import math

NUM_CLASSES=cfg.MNIST.num_classes
IMG_SIZE=cfg.MNIST.img_size



def inference(x_ph):
  img=tf.reshape(x_ph,[-1,IMG_SIZE,IMG_SIZE,1])
  #CONV1
  with tf.name_scope('conv1'):
    weights=tf.Variable(tf.truncated_normal([cfg.MNIST.LAYERS.CONV1.kernel_size,cfg.MNIST.LAYERS.CONV1.kernel_size,1,cfg.MNIST.LAYERS.CONV1.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
    bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.LAYERS.CONV1.num_channels]))
    conv1=tf.layers.conv2d(inputs=img,filters=cfg.MNIST.LAYERS.CONV1.num_channels, kernel_size=[cfg.MNIST.LAYERS.CONV1.kernel_size,cfg.MNIST.LAYERS.CONV1.kernel_size],padding='SAME',activation=tf.nn.relu)
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[cfg.MNIST.LAYERS.POOL1.kernel_size,cfg.MNIST.LAYERS.POOL1.kernel_size], strides=cfg.MNIST.LAYERS.POOL1.stride,name='pooling')

  #CONV2
  with tf.name_scope('conv2'):
    weights=tf.Variable(tf.truncated_normal([cfg.MNIST.LAYERS.CONV2.kernel_size,cfg.MNIST.LAYERS.CONV2.kernel_size,cfg.MNIST.LAYERS.CONV1.num_channels,cfg.MNIST.LAYERS.CONV2.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
    bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.LAYERS.CONV2.num_channels]))
    conv2=tf.layers.conv2d(inputs=pool1,filters=cfg.MNIST.LAYERS.CONV2.num_channels, kernel_size=[cfg.MNIST.LAYERS.CONV2.kernel_size,cfg.MNIST.LAYERS.CONV2.kernel_size],padding='SAME',activation=tf.nn.relu)
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[cfg.MNIST.LAYERS.POOL2.kernel_size,cfg.MNIST.LAYERS.POOL2.kernel_size], strides=cfg.MNIST.LAYERS.POOL2.stride)
    #pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]*pool2.shape[2]*cfg.MNIST.LAYERS.CONV2.num_channels])
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
  #FC1
  with tf.name_scope('fc1'):
    #weights=tf.Variable(tf.truncated_normal([pool2.shape[1]*pool2.shape[2]*cfg.MNIST.LAYERS.CONV2.num_channels,cfg.MNIST.LAYERS.FC1.size],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
    weights=tf.Variable(tf.truncated_normal([7*7*64,cfg.MNIST.LAYERS.FC1.size],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
    bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.LAYERS.FC1.size]))
    #fc1=tf.layers.dense(inputs=pool2_flat, units=cfg.MNIST.LAYERS.FC1.size, activation=tf.nn.relu, name='fc1')
    fc1=tf.add(tf.matmul(pool2_flat, weights),bias, name='fc1')
    fc1_do=tf.layers.dropout(inputs=fc1, rate=cfg.MNIST.LAYERS.FC1.dropout, training=True)
  #FC2
  with tf.name_scope('logits'):
    weights=tf.Variable(tf.truncated_normal([cfg.MNIST.LAYERS.FC1.size,NUM_CLASSES],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
    bias=tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES]))
    logits=tf.add(tf.matmul(fc1_do,weights),bias,name='logits')
    #logits=tf.layers.dense(inputs=fc1_do, units=NUM_CLASSES, name='op_to_run')
  return logits

def loss(logits, labels):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits),name='xentropy')
  return cross_entropy

def training(loss, learning_rate):
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op=optimizer.minimize(loss)
  return train_op

def evaluation(logits,labels,k=1):
  correct = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1),name='accuracy')
  return tf.reduce_mean(tf.cast(correct, tf.float32))
