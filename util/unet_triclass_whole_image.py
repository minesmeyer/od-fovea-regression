"""
The U-Net scaled for 512x512 inputs
"""

from keras.layers import Input, Concatenate, Dense
from keras.layers.core import Activation, Reshape, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import UpSampling2D

from keras import backend as K
from keras.optimizers import Adam

import numpy as np


class unet():

    def __init__(self, nch, sz=512, drop = 0.):

        self.sz = sz
        self.nch = nch
        self.nlayers = int(np.floor(np.log(sz)/np.log(2)))+1
        self.drop = drop

    # Define the neural network
    def get_unet(self, nf):

        input_n = Input((self.sz, self.sz, self.nch))

        ##### Encoder part
        conv1 = []
        # nch x 512 x 512
        conv = Conv2D(nf, (3, 3), padding='same')(input_n)
        conv = LeakyReLU(alpha=0.01)(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(self.drop)(conv)

        # nfxnch x 512 x 512
        conv1 = Conv2D(nf*2, (3, 3), padding='same', strides=(2, 2))(conv)
        # nfxnch x 256 x 256
        conv1 = LeakyReLU(alpha=0.01)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Dropout(self.drop)(conv1)

        # nfxnch x 256 x 256
        conv2 = Conv2D(nf*4, (3, 3), padding='same', strides=(2, 2))(conv1)
        # nfxnch x 128 x 128
        conv2 = LeakyReLU(alpha=0.01)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Dropout(self.drop)(conv2)

        # nfxnch x 128 x 128
        conv3 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(conv2)
        # nfxnch x 64 x 64
        conv3 = LeakyReLU(alpha=0.01)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Dropout(self.drop)(conv3)

        # nfxnch x 64 x 64
        conv4 = Conv2D(nf*8, (3, 3), padding='same', strides=(2, 2))(conv3)
        # nfxnch x 32 x 32
        conv4 = LeakyReLU(alpha=0.01)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Dropout(self.drop)(conv4)

        ##### Upsample path
        up4 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv4), conv3])
        # nfxnch x 64 x 64
        upconv1 = Conv2D(nf*8, (3, 3), padding='same')(up4)
        upconv1 = LeakyReLU(alpha=0.01)(upconv1)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Dropout(self.drop)(upconv1)

        upconv1 = Conv2D(nf*8, (3, 3), padding='same')(upconv1)
        upconv1 = LeakyReLU(alpha=0.01)(upconv1)
        upconv1 = BatchNormalization()(upconv1)
        upconv1 = Dropout(self.drop)(upconv1)

        up3 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(upconv1), conv2])
        # nfxnch x 128 x 128
        upconv2 = Conv2D(nf, (3, 3), padding='same')(up3)
        upconv2 = LeakyReLU(alpha=0.01)(upconv2)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Dropout(self.drop)(upconv2)

        upconv2 = Conv2D(nf, (3, 3), padding='same')(upconv2)
        upconv2 = LeakyReLU(alpha=0.01)(upconv2)
        upconv2 = BatchNormalization()(upconv2)
        upconv2 = Dropout(self.drop)(upconv2)

        up2 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(upconv2), conv1])
        # nfxnch x 256 x 256
        upconv3 = Conv2D(nf, (3, 3), padding='same')(up2)
        upconv3 = LeakyReLU(alpha=0.01)(upconv3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Dropout(self.drop)(upconv3)

        upconv3 = Conv2D(nf, (3, 3), padding='same')(upconv3)
        upconv3 = LeakyReLU(alpha=0.01)(upconv3)
        upconv3 = BatchNormalization()(upconv3)
        upconv3 = Dropout(self.drop)(upconv3)

        up1= Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(upconv3), conv])
        # nfxnch x 512 x 512
        upconv4 = Conv2D(nf, (3, 3), padding='same')(up1)
        upconv4 = LeakyReLU(alpha=0.01)(upconv4)
        upconv4 = BatchNormalization()(upconv4)
        upconv4 = Dropout(self.drop)(upconv4)

        upconv4 = Conv2D(nf, (3, 3), padding='same')(upconv4)
        upconv4 = LeakyReLU(alpha=0.01)(upconv4)
        upconv4 = BatchNormalization()(upconv4)
        upconv4 = Dropout(self.drop)(upconv4)

        conv_final = Conv2D(1, (3,3), padding='same')(upconv4)
        act = 'sigmoid'
        out_distances= Activation(act)(conv_final)

        model_distances = Model(inputs=input_n, outputs=out_distances)

        return model_distances
