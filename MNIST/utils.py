from setting import cfg
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import os
import pickle


def scatter(x, labels):
    if x.shape[-1]==2:
        vis_x = x[:, 0]
        vis_y = x[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", 10))
    else:
        vis_x = x[:, 0]
        vis_y = x[:, 1]
        vis_z = x[:, 2]
        plt.scatter(vis_x, vis_y, zs=vis_z, c=labels, cmap=plt.cm.get_cmap("jet", 10))
    #plt.show()
    return plt
##    plt.xlim(-25, 25)
##    plt.ylim(-25, 25)
##    ax.axis('off')
##    ax.axis('tight')


from tensorflow.examples.tutorials.mnist import input_data
mnist_db = input_data.read_data_sets('MNIST_data', one_hot=True)

images = mnist_db.test.images
labels = np.argmax(mnist_db.test.labels,axis=1)

x=np.array(images)
model = TSNE(n_components=2, verbose=1,n_iter=cfg.UTILS.TSNE.iters)
model.fit_transform(x)
model3d = TSNE(n_components=3, verbose=1,n_iter=10000)
model3d.fit_transform(x)
models_dict={'model':model.embedding_,'model3d':model3d.embedding_}
np.save(os.path.join(cfg.UTILS.TSNE.model_path,'model.npy'),model.embedding_)

vis=scatter(model.embedding_,labels)
#m=visuaize_tsne(images, labels)

