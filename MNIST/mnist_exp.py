from setting import cfg
import tensorflow as tf
import mnist
import os

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
    sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()
    x=graph.get_tensor_by_name('x_ph:0')
    y_=graph.get_tensor_by_name('y_ph:0')
    output = graph.get_tensor_by_name("logits/logits:0")

    return sess,x,y_,output

def adversarial_example(x, sess ,x_ph, logits, new_class_idx):
    output_dim=logits.shape[-1].value
    batch_size=x.shape[0]
    if new_class_idx>=output_dim:
        print "new class index exceeds output vector dimensions"
        return None
    indices=tf.constant(new_class_idx, shape=[batch_size])
    target = tf.one_hot(indices, output_dim)
    eps = -tf.abs(cfg.ADV.eps)
    for i in range(cfg.ADV.epochs):
        #output=mnist.inference(x_ph)
        sess.run(tf.global_variables_initializer())
        #prediction=sess.run(logits,feed_dict={x_ph:x})
        loss=mnist.loss(logits, target)
        dy_dx=tf.gradients(loss, x)
        print dy_dx
        grads= sess.run(dy_dx,feed_dict={x_ph:x})
        print dy_dx
        x = tf.stop_gradient(x + eps*tf.sign(dy_dx))
        x = tf.clip_by_value(x, cfg.ADV.min_grad_clip,cfg.ADV.max_grad_clip)
    return x
    
    
def dump_images():
    pass
    

#mnist_training()
sess ,x_ph, y_ph, output=load_model()
batch = mnist_db.train.next_batch(13)
adv=adversarial_example(batch[0], sess, x_ph, output, 3)


