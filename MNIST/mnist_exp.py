from setting import cfg
import tensorflow as tf
import mnist

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
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE*IMG_SIZE])
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
  
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
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            #targets=sess.run(tf.cast(mnist_db.test.labels,tf.int32))
            #prediction=sess.run(eval_correct,feed_dict={x:mnist_db.test.images,y_:targets})
            #print('Step %d: loss = %.2f, accuracy %.4f' % (i, loss_val,prediction))
            saver.save(sess,'model',global_step=i)
        batch = mnist_db.train.next_batch(BATCH_SIZE)
        _,loss_val=sess.run([train_op,loss],feed_dict={x: batch[0], y_: batch[1]})
        saver.save(sess,'final-model')

mnist_training()
