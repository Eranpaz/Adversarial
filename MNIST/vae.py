from setting import cfg
import tensorflow as tf
import math
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist_db = input_data.read_data_sets('MNIST_data', one_hot=True)

from tensorflow.examples.tutorials.mnist import input_data
mnist_db = input_data.read_data_sets('MNIST_data', one_hot=True)

NUM_CLASSES=cfg.MNIST.num_classes
IMG_SIZE=cfg.MNIST.img_size
BATCH_SIZE=cfg.MNIST.batch_size
LR=cfg.MNIST.lr

if cfg.MNIST.epochs==None:
    MAX_ITER=cfg.MNIST.max_iter
else:
    #need to calculate number of iterations based on epochs & batch size
    pass

def lr_sig(x,shift,mult):
    return 1/(1+math.exp(-(x+shift)*mult))

def encoder(x_ph):
    img=tf.reshape(x_ph,[-1,IMG_SIZE,IMG_SIZE,1])
    with tf.name_scope('enc_conv1'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.ENC.CONV1.kernel_size,cfg.MNIST.VAE.ENC.CONV1.kernel_size,1,cfg.MNIST.VAE.ENC.CONV1.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.CONV1.num_channels]))
        conv1=tf.layers.conv2d(inputs=img,filters=cfg.MNIST.VAE.ENC.CONV1.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV1.kernel_size,cfg.MNIST.VAE.ENC.CONV1.kernel_size],strides=cfg.MNIST.VAE.ENC.CONV1.stride,padding='SAME')
        bn1=tf.layers.batch_normalization(conv1, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn1=tf.nn.relu(bn1)
    with tf.name_scope('enc_conv2'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.ENC.CONV2.kernel_size,cfg.MNIST.VAE.ENC.CONV2.kernel_size,1,cfg.MNIST.VAE.ENC.CONV2.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.ENC.CONV1.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.CONV2.num_channels]))
        conv2=tf.layers.conv2d(inputs=bn1,filters=cfg.MNIST.VAE.ENC.CONV2.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV2.kernel_size,cfg.MNIST.VAE.ENC.CONV2.kernel_size],strides=cfg.MNIST.VAE.ENC.CONV2.stride,padding='SAME')
        bn2=tf.layers.batch_normalization(conv2, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn2=tf.nn.relu(bn2)
    with tf.name_scope('enc_conv3'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.ENC.CONV3.kernel_size,cfg.MNIST.VAE.ENC.CONV3.kernel_size,1,cfg.MNIST.VAE.ENC.CONV3.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.ENC.CONV2.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.CONV3.num_channels]))
        conv3=tf.layers.conv2d(inputs=bn2,filters=cfg.MNIST.VAE.ENC.CONV3.num_channels, kernel_size=[cfg.MNIST.VAE.ENC.CONV3.kernel_size,cfg.MNIST.VAE.ENC.CONV3.kernel_size],strides=cfg.MNIST.VAE.ENC.CONV3.stride,padding='SAME')
        bn3=tf.layers.batch_normalization(conv3, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn3=tf.nn.relu(bn3)
        flatten=tf.contrib.layers.flatten(bn3)
    
    with tf.name_scope('z_mean'):
        weights=tf.Variable(tf.truncated_normal([flatten.shape[1].value,cfg.MNIST.VAE.ENC.z_mean_size],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.ENC.CONV3.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.z_mean_size]))
        z_mean=tf.add(tf.matmul(flatten, weights),bias, name='z_mean')

    with tf.name_scope('z_var'):
        weights=tf.Variable(tf.truncated_normal([flatten.shape[1].value,cfg.MNIST.VAE.ENC.z_var_size],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.ENC.CONV3.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.ENC.z_var_size]))
        z_var=tf.add(tf.matmul(flatten, weights),bias, name='z_var')

    return z_mean, z_var

def generator(z_x_ph):
    with tf.name_scope('gen_fc'):
        weights=tf.Variable(tf.truncated_normal([z_x_ph.shape[1].value,cfg.MNIST.VAE.GEN.FC.size],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.ENC.z_var_size))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.FC.size]))
        fc=tf.add(tf.matmul(z_x_ph,weights),bias, name='fc')
        fc=tf.reshape(fc,(-1,7,7,256),name='fc_reshaped')
        bn4=tf.layers.batch_normalization(fc, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn4=tf.nn.relu(bn4,name='fc_reshaped')

    with tf.name_scope('gen_deconv1'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.GEN.DECONV1.kernel_size,cfg.MNIST.VAE.GEN.DECONV1.kernel_size,1,cfg.MNIST.VAE.GEN.DECONV1.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.GEN.FC.size))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.DECONV1.num_channels]))
        deconv1=tf.layers.conv2d_transpose(inputs=bn4,filters=cfg.MNIST.VAE.GEN.DECONV1.num_channels, kernel_size=[cfg.MNIST.VAE.GEN.DECONV1.kernel_size,cfg.MNIST.VAE.GEN.DECONV1.kernel_size],strides=cfg.MNIST.VAE.GEN.DECONV1.stride,padding='SAME')
        bn5=tf.layers.batch_normalization(deconv1, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn5=tf.nn.relu(bn5,name='deconv1')
        
    with tf.name_scope('gen_deconv2'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.GEN.DECONV2.kernel_size,cfg.MNIST.VAE.GEN.DECONV2.kernel_size,1,cfg.MNIST.VAE.GEN.DECONV2.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.GEN.DECONV1.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.DECONV2.num_channels]))
        deconv2=tf.layers.conv2d_transpose(inputs=bn5,filters=cfg.MNIST.VAE.GEN.DECONV2.num_channels, kernel_size=[cfg.MNIST.VAE.GEN.DECONV2.kernel_size,cfg.MNIST.VAE.GEN.DECONV2.kernel_size],strides=cfg.MNIST.VAE.GEN.DECONV2.stride,padding='SAME')
        bn6=tf.layers.batch_normalization(deconv2, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn6=tf.nn.relu(bn6,name='deconv2')

    with tf.name_scope('gen_deconv3'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.GEN.DECONV3.kernel_size,cfg.MNIST.VAE.GEN.DECONV3.kernel_size,1,cfg.MNIST.VAE.GEN.DECONV3.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.GEN.DECONV2.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.DECONV3.num_channels]))
        deconv3=tf.layers.conv2d_transpose(inputs=bn6,filters=cfg.MNIST.VAE.GEN.DECONV3.num_channels, kernel_size=[cfg.MNIST.VAE.GEN.DECONV3.kernel_size,cfg.MNIST.VAE.GEN.DECONV3.kernel_size],strides=cfg.MNIST.VAE.GEN.DECONV3.stride,padding='SAME')
        bn7=tf.layers.batch_normalization(deconv3, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn7=tf.nn.relu(bn7,name='deconv3')

    with tf.name_scope('gen_x'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.GEN.X.kernel_size,cfg.MNIST.VAE.GEN.X.kernel_size,1,cfg.MNIST.VAE.GEN.X.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.GEN.DECONV3.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.GEN.X.num_channels]))
        deconv4=tf.layers.conv2d_transpose(inputs=bn7,filters=cfg.MNIST.VAE.GEN.X.num_channels, kernel_size=[cfg.MNIST.VAE.GEN.X.kernel_size,cfg.MNIST.VAE.GEN.X.kernel_size], strides=cfg.MNIST.VAE.GEN.X.stride,padding='SAME')
        x_tilde=tf.nn.tanh(deconv4,name='deconv4')
        flatten=tf.contrib.layers.flatten(x_tilde)

    #return flatten
    return x_tilde
        

def discriminator(x_tilde_ph):
    #x_tilde=tf.identity(x_tilde_ph)
    x_tilde=tf.reshape(x_tilde_ph,[-1,IMG_SIZE,IMG_SIZE,1])
    with tf.name_scope('dis_conv1'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.DIS.CONV1.kernel_size,cfg.MNIST.VAE.DIS.CONV1.kernel_size,1,cfg.MNIST.VAE.DIS.CONV1.num_channels],stddev=1.0 / math.sqrt(float(IMG_SIZE*IMG_SIZE))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.DIS.CONV1.num_channels]))
        conv1=tf.layers.conv2d(inputs=x_tilde,filters=cfg.MNIST.VAE.DIS.CONV1.num_channels, kernel_size=[cfg.MNIST.VAE.DIS.CONV1.kernel_size,cfg.MNIST.VAE.DIS.CONV1.kernel_size],padding='SAME')
        conv1=tf.nn.relu(conv1)

    with tf.name_scope('dis_conv2'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.DIS.CONV2.kernel_size,cfg.MNIST.VAE.DIS.CONV2.kernel_size,1,cfg.MNIST.VAE.DIS.CONV2.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.DIS.CONV1.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.DIS.CONV2.num_channels]))
        conv2=tf.layers.conv2d(inputs=conv1,filters=cfg.MNIST.VAE.DIS.CONV2.num_channels, kernel_size=[cfg.MNIST.VAE.DIS.CONV2.kernel_size,cfg.MNIST.VAE.DIS.CONV2.kernel_size],padding='SAME')
        bn8=tf.layers.batch_normalization(conv2, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn8=tf.nn.relu(bn8)

    with tf.name_scope('dis_conv3'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.DIS.CONV3.kernel_size,cfg.MNIST.VAE.DIS.CONV3.kernel_size,1,cfg.MNIST.VAE.DIS.CONV3.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.DIS.CONV2.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.DIS.CONV3.num_channels]))
        conv3=tf.layers.conv2d(inputs=conv2,filters=cfg.MNIST.VAE.DIS.CONV3.num_channels, kernel_size=[cfg.MNIST.VAE.DIS.CONV3.kernel_size,cfg.MNIST.VAE.DIS.CONV3.kernel_size],padding='SAME')
        bn9=tf.layers.batch_normalization(conv3, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn9=tf.nn.relu(bn9)

    with tf.name_scope('dis_conv4'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.DIS.CONV4.kernel_size,cfg.MNIST.VAE.DIS.CONV4.kernel_size,1,cfg.MNIST.VAE.DIS.CONV4.num_channels],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.DIS.CONV3.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.DIS.CONV4.num_channels]))
        conv4=tf.layers.conv2d(inputs=conv3,filters=cfg.MNIST.VAE.DIS.CONV4.num_channels, kernel_size=[cfg.MNIST.VAE.DIS.CONV4.kernel_size,cfg.MNIST.VAE.DIS.CONV4.kernel_size],padding='SAME')
        bn10=tf.layers.batch_normalization(conv4, axis=cfg.MNIST.VAE.ENC.BN.axis, momentum=cfg.MNIST.VAE.ENC.BN.momentum, epsilon=cfg.MNIST.VAE.ENC.BN.eps,center=cfg.MNIST.VAE.ENC.BN.center, scale=cfg.MNIST.VAE.ENC.BN.scale)
        bn10=tf.nn.relu(bn10)
        flatten=tf.contrib.layers.flatten(bn10)

    with tf.name_scope('dis_fc'):
        weights=tf.Variable(tf.truncated_normal([flatten.shape[1].value,cfg.MNIST.VAE.DIS.FC.size],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.DIS.CONV4.num_channels))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[cfg.MNIST.VAE.DIS.FC.size]))
        fc=tf.add(tf.matmul(flatten, weights),bias)
        fc=tf.nn.relu(fc, name='fc')

    with tf.name_scope('dis_logits'):
        weights=tf.Variable(tf.truncated_normal([cfg.MNIST.VAE.DIS.FC.size,1],stddev=1.0 / math.sqrt(float(cfg.MNIST.VAE.DIS.FC.size))),name='weights')
        bias=tf.Variable(tf.constant(0.1,shape=[1]))
        logits=tf.add(tf.matmul(fc,weights),bias)
        D=tf.nn.sigmoid(logits,name='logits')

    return D,fc    
    

def inference(x_ph):
    z_p =  tf.random_normal((BATCH_SIZE,cfg.MNIST.VAE.ENC.z_mean_size ), 0, 1) # normal dist for GAN
    eps = tf.random_normal((BATCH_SIZE,cfg.MNIST.VAE.ENC.z_mean_size ), 0, 1) # normal dist for VAE

    with tf.variable_scope("enc"):
        z_mean, z_var =encoder(x_ph)
    with tf.variable_scope("gen"):
        z_x = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_var)), eps)) # grab our actual z
        print "z_x shape:", z_x.shape
        x_tilde = generator(z_x)
    with tf.variable_scope("dis"):
        _, l_x_tilde = discriminator(x_tilde)

    with tf.variable_scope("gen", reuse=True):         
        x_p = generator(z_p)    
    with tf.variable_scope("dis", reuse=True):
        d_x, l_x = discriminator(x_ph)  # positive examples              
    with tf.variable_scope("dis", reuse=True):
        d_x_p, _ = discriminator(x_p)  
    return z_mean, z_var, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p


    

def loss(x_ph, x_tilde, z_var, z_mean, d_x, d_x_p, l_x, l_x_tilde):
    
    """
    Loss functions for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    img=tf.reshape(x_ph,[-1,IMG_SIZE,IMG_SIZE,1])
    ### We don't actually use SSE (MSE) loss for anything (but maybe pretraining)
    SSE_loss = tf.reduce_mean(tf.square(img - x_tilde)) # This is what a normal VAE uses

    # We clip gradients of KL divergence to prevent NANs
    KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_var, -10.0, 10.0)- tf.square(tf.clip_by_value(z_mean, -10.0, 10.0) ) - tf.exp(tf.clip_by_value(z_var, -10.0, 10.0) ), 1))/IMG_SIZE/IMG_SIZE
    # Discriminator Loss
    D_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(d_x,1e-5,1.0)) + tf.log(tf.clip_by_value(1.0 - d_x_p,1e-5,1.0))))
    # Generator Loss    
    G_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(d_x_p,1e-5,1.0))))# + 
                                    #tf.log(tf.clip_by_value(1.0 - d_x,1e-5,1.0))))
    # Lth Layer Loss - the 'learned similarity measure'  
    LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde))/IMG_SIZE/IMG_SIZE
    return SSE_loss, KL_loss, D_loss, G_loss, LL_loss


def training(SSE_loss, KL_loss, D_loss, G_loss, LL_loss, lr_e_ph, lr_g_ph, lr_d_ph):
    L_e = tf.clip_by_value(KL_loss*cfg.MNIST.VAE.LOSSES.KL_weight + LL_loss, -100, 100)
    L_g = tf.clip_by_value(LL_loss*cfg.MNIST.VAE.LOSSES.LL_weight+G_loss*cfg.MNIST.VAE.LOSSES.G_weight, -100, 100)
    L_d = tf.clip_by_value(D_loss, -100, 100)

    trainables = tf.trainable_variables()
    e_vars = [v for v in trainables if 'enc' in v.name]
    print "e_vars:",len(e_vars)
    g_vars = [v for v in trainables if 'gen' in v.name]
    print "g_vars:",len(g_vars)
    d_vars = [v for v in trainables if 'dis' in v.name]
    print "d_vars:",len(d_vars)
    
    opt_e = tf.train.AdamOptimizer(lr_e_ph,epsilon=1.)
    opt_g = tf.train.AdamOptimizer(lr_g_ph,epsilon=1.)
    opt_d = tf.train.AdamOptimizer(lr_d_ph,epsilon=1.)

    train_e = opt_e.minimize(L_e, var_list=e_vars)
    train_g = opt_g.minimize(L_g, var_list=g_vars)
    train_d = opt_d.minimize(L_d, var_list=d_vars)

    return train_e,train_g, train_d


def train():
    x_ph = tf.placeholder(tf.float32, shape=[None, IMG_SIZE*IMG_SIZE], name='x_ph')
    lr_e_ph=tf.placeholder(tf.float32,shape=[])
    lr_g_ph=tf.placeholder(tf.float32,shape=[])
    lr_d_ph=tf.placeholder(tf.float32,shape=[])
    z_mean, z_var, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p=inference(x_ph)
    SSE_loss, KL_loss, D_loss, G_loss, LL_loss=loss(x_ph, x_tilde, z_var, z_mean, d_x, d_x_p, l_x, l_x_tilde)
    train_e,train_g, train_d=training(SSE_loss, KL_loss, D_loss, G_loss, LL_loss,lr_e_ph,lr_g_ph,lr_d_ph)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=cfg.MNIST.RUN.models_to_save)
    sess = tf.Session()
    sess.run(init)

    print "*****TRAINING STARTED*******"
    d_fake=0.5
    d_real=0.5
    lr_summary=[]
    loss_summary=[]
    for i in range(MAX_ITER):
        print i
        if i%100==0 and i>0:
            print('Step %d: Discriminator loss = %.5f, Generator loss = %.5f Similarity loss = %.7f' % (i,D_err, G_err, LL_err ))
            print("classification errors: d_fake=%.7f d_real=%.7f" %(np.mean(d_fake), np.mean(d_real)))
            print('Current LR: e_lr=%.7f g_lr=%.7f d_lr=%.7f' % (lr_e,lr_g,lr_d))
            loss_summary.append((D_err, G_err, KL_err, SSE_err, LL_err, np.mean(d_fake),np.mean(d_real)))
            lr_summary.append((lr_e,lr_g,lr_d))
        if i%1000==0 and i>0:
            saver.save(sess,os.path.join(cfg.MNIST.RUN.models_dir,'model_vae'),global_step=i)
            summaries={'loss':loss_summary,'lr':lr_summary}
            fout=open('./models/summaries.pkl','w')
            pickle.dump(summaries,fout)
            fout.close()
        batch = mnist_db.train.next_batch(BATCH_SIZE)
        lr_e=LR*lr_sig(np.mean(d_real),-0.5,15)
        lr_g=LR*lr_sig(np.mean(d_real),-0.5,15)
        lr_d=LR*lr_sig(np.mean(d_fake),-0.5,15)
        _, _, _, D_err, G_err, KL_err, SSE_err, LL_err, d_fake,d_real = sess.run([train_e,train_g,train_d,D_loss, G_loss, KL_loss, SSE_loss, LL_loss,d_x_p, d_x],feed_dict={x_ph:batch[0],lr_e_ph:lr_e, lr_g_ph:lr_g, lr_d_ph:lr_d})
        #print('Step %d: Discriminator loss = %.2f, Generator loss = %.2f' % (i,D_err, G_err ))
    #saver.save(sess,os.path.join(cfg.MNIST.RUN.models_dir,cfg.MNIST.RUN.last_model_name))
    

def generate():
    sess=tf.Session()
    saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(os.path.join(cfg.MNIST.RUN.models_dir))+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(cfg.MNIST.RUN.models_dir)))
    graph = tf.get_default_graph()
    z_ph =tf.placeholder(tf.float32)
    generated=graph.get_tensor_by_name('gen/gen_x/deconv4:0')
    
    x_ph=graph.get_tensor_by_name('x_ph:0')
    
    
    batch = mnist_db.train.next_batch(BATCH_SIZE)
    sess.run(tf.global_variables_initializer())
    z_p =  sess.run(tf.random_normal((BATCH_SIZE,cfg.MNIST.VAE.ENC.z_mean_size ), 0, 1))
    output=sess.run(generated, feed_dict={x_ph:batch[0]})
    return output

def visualize_learning():
    fin=open('./models/summaries.pkl','r')
    data=pickle.load(fin)
    fin.close()

    lr=data['lr']
    loss=data['loss']

    D_loss=[]
    G_loss=[]
    KL_loss=[]
    LL_loss=[]
    d_fake=[]
    d_real=[]
    for (d,g,k,_,l,f,r) in loss:
        D_loss.append(d)
        G_loss.append(g)
        KL_loss.append(k)
        LL_loss.append(l)
        d_fake.append(f)
        d_real.append(r)
    lr_e=[]
    lr_g=[]
    lr_d=[]
    for (e,g,d) in lr:
        lr_e.append(e)
        lr_g.append(g)
        lr_d.append(d)

    x=t = np.arange(len(D_loss))
    plt.subplot(311)
    plt.plot(x,D_loss,'r-', label='D_loss')
    plt.plot(x,G_loss,'g-',label='G_loss')
    plt.plot(x,KL_loss,'b-',label='KL_loss')
    #plt.plot(x,LL_loss,'k-',label='LL_loss')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.subplot(312)
    plt.plot(x,lr_e,'r-',label='encoder\generator_lr')
    #plt.plot(x,lr_g,'g-',label='generator_lr')
    plt.plot(x,lr_d,'b-',label='discriminator_lr')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.yscale('log')
    plt.ylim((1e-7,1))
    plt.subplot(313)
    plt.plot(x,d_real,'r-',label='d_real')
    plt.plot(x,d_fake,'b-',label='d_fake')
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.show()


#train()
#g=generate()
visualize_learning()
