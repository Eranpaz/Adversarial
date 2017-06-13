from setting import cfg
import tensorflow as tf
import mnist
import os
import numpy as np
from scipy import misc
import random


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

def iterator(images, labels):
    if np.ndim(images)==3:
        images=images.reshape(images.shape[1],images.shape[2])
    num_batches=int(np.ceil(images.shape[0]/float(BATCH_SIZE)))
    #print "required number of bataches is:",num_batches
    img_list=[]
    lbl_list=[]
    for i in range(num_batches):
        img_list.append(images[i*BATCH_SIZE:min((i+1)*BATCH_SIZE,images.shape[0]),:])
        lbl_list.append(labels[i*BATCH_SIZE:min((i+1)*BATCH_SIZE,labels.shape[0])])
    return img_list,lbl_list


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

    graph = tf.get_default_graph()
    x=graph.get_tensor_by_name('x_ph:0')
    y_=graph.get_tensor_by_name('y_ph:0')
    fc1=graph.get_tensor_by_name('fc1/fc1:0')
    output = graph.get_tensor_by_name("logits/logits:0")
    loss=graph.get_tensor_by_name("xentropy:0")
    correct=graph.get_tensor_by_name("accuracy:0")

    return sess,x,y_,fc1,output,loss, correct

##def adversarial_example(x, y, x_ph, y_ph, logits, new_class_idx):
##    eps = -tf.abs(cfg.ADV.eps)
##    adv_images=None
##    
##    output_dim=logits.shape[-1].value
##    input_size=x.shape[0]
##    if new_class_idx>=output_dim:
##        print "new class index exceeds output vector dimensions"
##        return None
##    indices=tf.constant(new_class_idx, shape=[input_size])
##    one_hot = sess.run(tf.one_hot(indices, output_dim))
##    img_list, lbl_list=iterator(x,one_hot)
##    print ("running for %d epochs" %cfg.ADV.epochs)
##    for b in range(len(img_list)):
##        if b%10==0:
##            print ("running batch number %d" %b)
##        for i in range(cfg.ADV.epochs):
##            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lbl_list[b], logits=logits))
##            dy_dx=tf.gradients(loss, x_ph)
##            imgs=sess.run(tf.identity(img_list[b]))
##            grads = tf.stop_gradient(imgs + eps*tf.sign(dy_dx))
##            x_new,l,g = sess.run([tf.clip_by_value(grads, cfg.ADV.min_grad_clip,cfg.ADV.max_grad_clip),loss, dy_dx],feed_dict={x_ph:imgs,y_ph:lbl_list[b]})
##            x_new=x_new.reshape(x_new.shape[1],x_new.shape[2])
##        if type(adv_images) is not np.ndarray:
##            adv_images=x_new
##        else:
##            adv_images=np.concatenate((adv_images, x_new))
##    return adv_images,l,g

def adversarial_example(x, y, x_ph, y_ph, logits, new_class_idx):
    eps = -tf.abs(cfg.ADV.eps)
    adv_images=None
    
    output_dim=logits.shape[-1].value
    input_size=x.shape[0]
    if new_class_idx>=output_dim:
        print "new class index exceeds output vector dimensions"
        return None
    indices=tf.constant(new_class_idx, shape=[input_size])
    one_hot = sess.run(tf.one_hot(indices, output_dim))
    img_list, lbl_list=iterator(x,one_hot)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_ph, logits=logits))
    dy_dx=tf.gradients(loss, x_ph)
    x_new = tf.stop_gradient(x_ph + eps*tf.sign(dy_dx))
    x_new = tf.clip_by_value(x_new, cfg.ADV.min_grad_clip,cfg.ADV.max_grad_clip)
    
    print ("running for %d epochs" %cfg.ADV.epochs)
    for b in range(len(img_list)):
        if b%10==0:
            print ("running batch number %d" %b)
        for i in range(cfg.ADV.epochs):
            if i==0:
                result=sess.run(x_new, feed_dict={x_ph:img_list[b],y_ph:lbl_list[b]})
                result=np.reshape(result, (result.shape[1],result.shape[2]))
            else:
                tmp=sess.run(x_new, feed_dict={x_ph:result,y_ph:lbl_list[b]})
                result=np.reshape(tmp, (tmp.shape[1],tmp.shape[2]))
        if type(adv_images) is not np.ndarray:
            adv_images=result
        else:
            adv_images=np.concatenate((adv_images, result))
    return adv_images

    
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
    img_list, lbl_list=iterator(images, labels)
    acc=[]
    for i in range(len(img_list)):
        p=sess.run(correct,feed_dict={x_ph:img_list[i],y_ph:lbl_list[i]})
        acc.extend(p)
    return sess.run(tf.reduce_mean(tf.cast(acc,tf.float32)))



    
        

#mnist_training()

sess ,x_ph,y_ph,fc1,output,loss,correct=load_model()
#images,labels = mnist_db.test.next_batch(128)
#new_images, loss, grads=adversarial_example(images,labels, x_ph,y_ph, output, 5)
orig_imgs=mnist_db.test.images
labels=mnist_db.test.labels
idx=np.where(np.argmax(labels,axis=1)==1)[0]
random.shuffle(idx)
sub_labels=labels[idx[:200]]
sub_images=orig_imgs[idx[:200]]
l=np.zeros((200,10))
l[:,5]=1
for i in range(1):
    cfg.ADV.epochs=10
    new_images=adversarial_example(sub_images, sub_labels, x_ph, y_ph, output, 5)
    print "returned images shape:", new_images.shape
    print ("for %d epochs accuraccy is %f" %(i,eval_acc(new_images, sub_labels, x_ph, y_ph)))
    print ("for %d epochs accuraccy is %f" %(i,eval_acc(new_images, l, x_ph, y_ph)))
#print eval_acc(b,l,x_ph,y_ph)
#dump_images(mnist_db.train.images, mnist_db.train.labels)
#new_images, loss, grads=adversarial_example(images,labels, x_ph,y_ph, output, 5)
#print eval_acc(new_images,labels,x_ph,y_ph)
