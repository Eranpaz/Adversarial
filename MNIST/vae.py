from setting import cfg
import tensorflow as tf
import math

NUM_CLASSES=cfg.MNIST.num_classes
IMG_SIZE=cfg.MNIST.img_size

def encoder(x_ph):
    img=tf.reshape(x_ph,[-1,IMG_SIZE,IMG_SIZE,1])
    with tf.name_scope('enc_conv1'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.ENC.CONV1.kernel_size,cfg.MNIST.VAE.ENC.CONV1.kernel_size,1,cfg.MNIST.VAE.ENC.CONV1.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.CONV1.num_channels]))
        conv1=tf.layers.conv2d(inputs=img,filters=cfg.MNIST.VAE.ENC.CONV1.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV1.kernel_size,cfg.MNIST.VAE.ENC.CONV1.kernel_size],padding='SAME')
        bn1=tf.layers.batch_normalization(conv1, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn1=tf.nn.relu(bn1)
    with tf.name_scope('enc_conv2'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.ENC.CONV2.kernel_size,cfg.MNIST.VAE.ENC.CONV2.kernel_size,1,cfg.MNIST.VAE.ENC.CONV2.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.CONV2.num_channels]))
        conv2=tf.layers.conv2d(inputs=bn1,filters=cfg.MNIST.VAE.ENC.CONV2.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV2.kernel_size,cfg.MNIST.VAE.ENC.CONV2.kernel_size],padding='SAME')
        bn2=tf.layers.batch_normalization(conv2, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn2=tf.nn.relu(bn2)
    with tf.name_scope('enc_conv3'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.ENC.CONV3.kernel_size,cfg.MNIST.VAE.ENC.CONV3.kernel_size,1,cfg.MNIST.VAE.ENC.CONV3.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.CONV3.num_channels]))
        conv3=tf.layers.conv2d(inputs=bn2,filters=cfg.MNIST.VAE.ENC.CONV3.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV3.kernel_size,cfg.MNIST.VAE.ENC.CONV3.kernel_size],padding='SAME')
        bn3=tf.layers.batch_normalization(conv2, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn3=tf.nn.relu(bn3)
        flatten=tf.contrib.layers.flatten(bn3)
    
    with tf.name_scope('z_mean'):
        weights=tf.Variable(tf.truncated_normal([flatten.shape[1].value,cfg.MNIST.VAE.ENC.z_mean_size],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.z_mean_size]))
        z_mean=tf.add(tf.matmul(flatten, weights),bias, name='z_mean')

    with tf.name_scope('z_var'):
        weights=tf.Variable(tf.truncated_normal([flatten.shape[1].value,cfg.MNIST.VAE.ENC.z_var_size],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.z_var_size]))
        z_var=tf.add(tf.matmul(flatten, weights),bias, name='z_var')

    return z_mean, z_var

def generator(z_ph):
    with tf.name_scope('gen_fc'):
        weights=tf.Variable(tf.truncated_normal([z_ph.shape[1].value,cfg.MNIST.VAE.GEN.FC.size],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.FC.size]))
        fc=tf.add(tf.matmul(z_ph,weights),bias, name='fc')
        fc=tf.reshape(fc,(-1,8,8,256),name='fc_reshaped')
        bn4=tf.layers.batch_normalization(fc, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn4=tf.nn.relu(bn4,name='fc_reshaped')
    
    with tf.name_scope('gen_deconv1'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.GEN.DECONV1.kernel_size,cfg.MNIST.VAE.GEN.DECONV1.kernel_size,1,cfg.MNIST.VAE.GEN.DECONV1.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.DECONV1.num_channels]))
        deconv1=tf.layers.conv2d(inputs=fc,filters=cfg.MNIST.VAE.ENC.CONV1.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV1.kernel_size,cfg.MNIST.VAE.ENC.CONV1.kernel_size],padding='SAME')
        

def discriminator():
    pass

def inference():
    pass

def loss():
    pass
