import tensorflow as tf
from tensorflow.keras import layers

import os
from model.loss import *
import os
import datetime
from utils.utils import instantiate_from_config
from model.unet import *
from model.layers import *
import numpy as np
import tensorflow.keras.backend as K



class UnetSpatial_AutoEncoder(tf.keras.layers.Layer):


    def __init__(self,hidden_sizes,unet_num_res_blocks,drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_hidden_sizes=hidden_sizes
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("softmax")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//4, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.encoder_blocks.append(x)

                if hidden_size in self.unet_hidden_sizes[3:]:
                  idx_x=idx_x+1
                  x=SpatialTransformer(hidden_size//4)
                  self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

        self.middle_block=SpatialTransformer(self.unet_hidden_sizes[-1]//4)

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                if hidden_size in self.unet_hidden_sizes[3:]:
                  idx_x=idx_x+1
                  x=SpatialTransformer(hidden_size//4)
                  self.decoder_blocks.append(x)
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.pyramids= []         
        for hidden_size in range(self.unet_hidden_sizes[2:]):         
            x=DilatedSpatialPyramidPooling(hidden_size//2)
            self.pyramids.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(5, kernel_size=1,padding="same",name="classification_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor,context_tensor:tf.Tensor,hidden_sizes):
        x=self.conv_first(input_tensor)
        skips=[x]
        curr=0
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_size=self.pyramids[curr](hidden_sizes[curr])
                x = tf.concat([x, hidden_size], axis=-1)
                curr=curr+1
            if block.__class__.__name__=="SpatialTransformer":
                x=block([x,context_tensor])
            else:
                x=block(x)

            skips.append(x)

        x=self.middle_block([x,context_tensor])
        x = tf.concat([x, hidden_sizes[curr]], axis=-1)

        for idx,block in enumerate(self.decoder_blocks):
            if block.__class__.__name__=="SpatialTransformer":
                x=block([x,context_tensor])
            else:
                x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x

