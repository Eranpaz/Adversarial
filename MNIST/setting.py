import easydict
import os

cfg=easydict.EasyDict()

cfg.MNIST=easydict.EasyDict()
##MNIST DEFINITIONS
cfg.MNIST.num_classes=10
cfg.MNIST.img_size=28
cfg.MNIST.batch_size=128
cfg.MNIST.max_iter=20000
cfg.MNIST.epochs=None
cfg.MNIST.lr=1e-4

cfg.MNIST.RUN=easydict.EasyDict()
cfg.MNIST.RUN.root_dir='/home/puzi/thesis/MNIST'
cfg.MNIST.RUN.models_dir=os.path.join(cfg.MNIST.RUN.root_dir,'models')
cfg.MNIST.RUN.models_to_save=2
cfg.MNIST.RUN.last_model_name='final-model'
cfg.MNIST.RUN.dump_dir=os.path.join(cfg.MNIST.RUN.root_dir,'adv_images')

##MNIST LAYER DEFINITION
cfg.MNIST.LAYERS=easydict.EasyDict()
cfg.MNIST.LAYERS.CONV1=easydict.EasyDict()
cfg.MNIST.LAYERS.CONV1.num_channels=32
cfg.MNIST.LAYERS.CONV1.kernel_size=5
cfg.MNIST.LAYERS.CONV1.stride=1

cfg.MNIST.LAYERS.POOL1=easydict.EasyDict()
cfg.MNIST.LAYERS.POOL1.kernel_size=2
cfg.MNIST.LAYERS.POOL1.stride=2

cfg.MNIST.LAYERS.CONV2=easydict.EasyDict()
cfg.MNIST.LAYERS.CONV2.num_channels=64
cfg.MNIST.LAYERS.CONV2.kernel_size=5
cfg.MNIST.LAYERS.CONV2.stride=1

cfg.MNIST.LAYERS.POOL2=easydict.EasyDict()
cfg.MNIST.LAYERS.POOL2.kernel_size=2
cfg.MNIST.LAYERS.POOL2.stride=2

cfg.MNIST.LAYERS.FC1=easydict.EasyDict()
cfg.MNIST.LAYERS.FC1.size=1024
cfg.MNIST.LAYERS.FC1.dropout=0.5

##ADVERSARIAL EXAMPLES DEFINITION
cfg.ADV=easydict.EasyDict()
cfg.ADV.eps=0.02
cfg.ADV.epochs=10
cfg.ADV.min_grad_clip=0.
cfg.ADV.max_grad_clip=1.

##UTILS DEFINITION
cfg.UTILS=easydict.EasyDict()
##TSNE DEFINITION
cfg.UTILS.TSNE=easydict.EasyDict()
cfg.UTILS.TSNE.ndim=2
cfg.UTILS.TSNE.iters=10000
cfg.UTILS.TSNE.model_path=os.path.join(cfg.MNIST.RUN.root_dir,'models')
