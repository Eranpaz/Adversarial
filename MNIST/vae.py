from setting import cfg
import tensorflow as tf

def encoder():
    img=tf.reshape(x_ph,[-1,IMG_SIZE,IMG_SIZE,1])
    with tf.name_scope('enc_conv1'):
    weights=tf.Variable(tf.truncated_normal([cfg.MNIST.LAYERS.CONV1.kernel_size,cfg.MNIST.LAYERS.CONV1.kernel_size,1,cfg.MNIST.LAYERS.CONV1.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
    bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.LAYERS.CONV1.num_channels]))
    conv1=tf.layers.conv2d(inputs=img,filters=cfg.MNIST.LAYERS.CONV1.num_channels, kernel_size=[cfg.MNIST.LAYERS.CONV1.kernel_size,cfg.MNIST.LAYERS.CONV1.kernel_size],padding='SAME',activation=tf.nn.relu)
        

def generator():
    pass

def discriminator():
    pass

def inference():
    pass

def loss():
    pass
