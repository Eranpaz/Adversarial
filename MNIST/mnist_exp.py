from setting import cfg
import tensorflow as tf
import mnist
import os
import numpy as np
from scipy import misc


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



def mnist_training():
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE*IMG_SIZE], name='x_ph')
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_ph')
  
    logits = mnist.inference(x)
    loss = mnist.loss(logits, y_)
    train_op = mnist.training(loss, LR)
    eval_correct = mnist.evaluation(logits, y_)
    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=cfg.MNIST.RUN.models_to_save)
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    #summary_writer = tf.summary.FileWriter(, sess.graph)

    sess.run(init)
    print "*****TRAINING STARTED*******"
    for i in range(MAX_ITER):
        if i%100==0 and i>0:
            print('Step %d: loss = %.2f' % (i, loss_val))
            #targets=sess.run(tf.cast(mnist_db.test.labels,tf.int32))
            #prediction=sess.run(eval_correct,feed_dict={x:mnist_db.test.images,y_:targets})
            #print('Step %d: loss = %.2f, accuracy %.4f' % (i, loss_val,prediction))
            saver.save(sess,os.path.join(cfg.MNIST.RUN.models_dir,'model'),global_step=i)
        batch = mnist_db.train.next_batch(BATCH_SIZE)
        _,loss_val=sess.run([train_op,loss],feed_dict={x: batch[0], y_: batch[1]})
    saver.save(sess,os.path.join(cfg.MNIST.RUN.models_dir,cfg.MNIST.RUN.last_model_name))

def load_model():
    sess=tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(cfg.MNIST.RUN.models_dir,cfg.MNIST.RUN.last_model_name+'.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(os.path.join(cfg.MNIST.RUN.models_dir)))
    #sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()
    x=graph.get_tensor_by_name('x_ph:0')
    y_=graph.get_tensor_by_name('y_ph:0')
    output = graph.get_tensor_by_name("logits/logits:0")
    loss=graph.get_tensor_by_name("xentropy:0")
    correct=graph.get_tensor_by_name("accuracy:0")

    return sess,x,y_,output,loss, correct

def adversarial_example(x, y, x_ph, y_ph, logits, new_class_idx):
    #x,_ = mnist_db.train.next_batch(13)
    output_dim=logits.shape[-1].value
    batch_size=x.shape[0]
    if new_class_idx>=output_dim:
        print "new class index exceeds output vector dimensions"
        return None
    indices=tf.constant(new_class_idx, shape=[batch_size])
    target = sess.run(tf.one_hot(indices, output_dim))
    eps = -tf.abs(cfg.ADV.eps)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits))
    dy_dx=tf.gradients(loss, x_ph)
    x_new=tf.identity(x)
    for i in range(cfg.ADV.epochs):
        x_new = tf.stop_gradient(x_new + eps*tf.sign(dy_dx))
        x_new,g,l = sess.run([tf.clip_by_value(x_new, cfg.ADV.min_grad_clip,cfg.ADV.max_grad_clip),loss, dy_dx],feed_dict={x_ph:x,y_ph:target})
    return x_new,g,l

    
def dump_images(images, labels, adv_labels=[]):
    if np.ndim(images)==3:
        images=images.reshape(images.shape[1],images.shape[2])
    images=images.reshape(images.shape[0],int(np.sqrt(images.shape[1])),int(np.sqrt(images.shape[1])))
    print ("saving images to %s" %cfg.MNIST.RUN.dump_dir)
    if adv_labels:
        for i in range(images.shape[0]):    
                file_name=str(i)+'_'+str(np.argmax(labels[i]))+'_'+str(adv_labels[i])+'.jpg'
                misc.imsave(os.path.join(cfg.MNIST.RUN.dump_dir,file_name),images[i])
    else:
        for i in range(images.shape[0]):    
                file_name=str(i)+'_'+str(np.argmax(labels[i]))+'.jpg'
                misc.imsave(os.path.join(cfg.MNIST.RUN.dump_dir,file_name),images[i])
        
        

def eval_acc(images, labels, x_ph, y_ph):
    if np.ndim(images)==3:
        images=images.reshape(images.shape[1],images.shape[2])
    acc=sess.run(correct,feed_dict={x_ph:images,y_ph:labels})
    return sess.run(tf.reduce_mean(tf.cast(acc,tf.float32)))


    
        

#mnist_training()
sess ,x_ph, y_ph, output,loss,correct=load_model()
images,labels = mnist_db.test.next_batch(128)
#dump_images(mnist_db.train.images, mnist_db.train.labels)
new_images, loss, grads=adversarial_example(images,labels, x_ph,y_ph, output, 5)
print eval_acc(new_images,labels,x_ph,y_ph)
