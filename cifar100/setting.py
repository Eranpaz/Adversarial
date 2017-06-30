import easydict
import os

cfg=easydict.EasyDict()

cfg.CIFAR=easydict.EasyDict()
##CIFAR DEFINITIONS
cfg.CIFAR.format='MATRIX'
cfg.CIFAR.num_classes=100
cfg.CIFAR.batch_size=32
cfg.CIFAR.img_size=32

cfg.CIFAR.RUN=easydict.EasyDict()
cfg.CIFAR.RUN.root_dir='/home/puzi/thesis/cifar100'
cfg.CIFAR.RUN.models_dir=os.path.join(cfg.CIFAR.RUN.root_dir,'models')
cfg.CIFAR.RUN.models_to_save=2
cfg.CIFAR.RUN.last_model_name='final-model'
cfg.CIFAR.RUN.dump_dir=os.path.join(cfg.CIFAR.RUN.root_dir,'adv_images')


##CIFAR VAE CONFIGURATION
cfg.CIFAR.VAE=easydict.EasyDict()
cfg.CIFAR.VAE.ENC=easydict.EasyDict()

cfg.CIFAR.VAE.ENC.CONV1=easydict.EasyDict()
cfg.CIFAR.VAE.ENC.CONV1.num_channels=64
cfg.CIFAR.VAE.ENC.CONV1.kernel_size=5
cfg.CIFAR.VAE.ENC.CONV1.stride=2

cfg.CIFAR.VAE.ENC.CONV2=easydict.EasyDict()
cfg.CIFAR.VAE.ENC.CONV2.num_channels=128
cfg.CIFAR.VAE.ENC.CONV2.kernel_size=5
cfg.CIFAR.VAE.ENC.CONV2.stride=2

cfg.CIFAR.VAE.ENC.CONV3=easydict.EasyDict()
cfg.CIFAR.VAE.ENC.CONV3.num_channels=256
cfg.CIFAR.VAE.ENC.CONV3.kernel_size=5
cfg.CIFAR.VAE.ENC.CONV3.stride=1

cfg.CIFAR.VAE.ENC.z_mean_size=1024
cfg.CIFAR.VAE.ENC.z_var_size=1024

cfg.CIFAR.VAE.ENC.BN=easydict.EasyDict()
cfg.CIFAR.VAE.ENC.BN.momentum=0.9
cfg.CIFAR.VAE.ENC.BN.axis=1
cfg.CIFAR.VAE.ENC.BN.eps=1e-5
cfg.CIFAR.VAE.ENC.BN.center=False
cfg.CIFAR.VAE.ENC.BN.scale=False

cfg.CIFAR.VAE.GEN=easydict.EasyDict()
cfg.CIFAR.VAE.GEN.FC=easydict.EasyDict()
cfg.CIFAR.VAE.GEN.FC.size=8*8*256

cfg.CIFAR.VAE.GEN.DECONV1=easydict.EasyDict()
cfg.CIFAR.VAE.GEN.DECONV1.num_channels=256
cfg.CIFAR.VAE.GEN.DECONV1.kernel_size=5
cfg.CIFAR.VAE.GEN.DECONV1.stride=2

cfg.CIFAR.VAE.GEN.DECONV2=easydict.EasyDict()
cfg.CIFAR.VAE.GEN.DECONV2.num_channels=128
cfg.CIFAR.VAE.GEN.DECONV2.kernel_size=5
cfg.CIFAR.VAE.GEN.DECONV2.stride=2

cfg.CIFAR.VAE.GEN.DECONV3=easydict.EasyDict()
cfg.CIFAR.VAE.GEN.DECONV3.num_channels=32
cfg.CIFAR.VAE.GEN.DECONV3.kernel_size=5
cfg.CIFAR.VAE.GEN.DECONV3.stride=1

cfg.CIFAR.VAE.GEN.X=easydict.EasyDict()
cfg.CIFAR.VAE.GEN.X.num_channels=3
cfg.CIFAR.VAE.GEN.X.kernel_size=5
cfg.CIFAR.VAE.GEN.X.stride=1

cfg.CIFAR.VAE.DIS=easydict.EasyDict()
cfg.CIFAR.VAE.DIS.CONV1=easydict.EasyDict()
cfg.CIFAR.VAE.DIS.CONV1.num_channels=32
cfg.CIFAR.VAE.DIS.CONV1.kernel_size=5
cfg.CIFAR.VAE.DIS.CONV1.stride=2

cfg.CIFAR.VAE.DIS.CONV2=easydict.EasyDict()
cfg.CIFAR.VAE.DIS.CONV2.num_channels=128
cfg.CIFAR.VAE.DIS.CONV2.kernel_size=5
cfg.CIFAR.VAE.DIS.CONV2.stride=2

cfg.CIFAR.VAE.DIS.CONV3=easydict.EasyDict()
cfg.CIFAR.VAE.DIS.CONV3.num_channels=256
cfg.CIFAR.VAE.DIS.CONV3.kernel_size=5
cfg.CIFAR.VAE.DIS.CONV3.stride=2

cfg.CIFAR.VAE.DIS.CONV4=easydict.EasyDict()
cfg.CIFAR.VAE.DIS.CONV4.num_channels=256
cfg.CIFAR.VAE.DIS.CONV4.kernel_size=5
cfg.CIFAR.VAE.DIS.CONV4.stride=2

cfg.CIFAR.VAE.DIS.FC=easydict.EasyDict()
cfg.CIFAR.VAE.DIS.FC.size=512

cfg.CIFAR.VAE.LOSSES=easydict.EasyDict()
cfg.CIFAR.VAE.LOSSES.KL_weight=1
cfg.CIFAR.VAE.LOSSES.LL_weight=2
cfg.CIFAR.VAE.LOSSES.G_weight=2
