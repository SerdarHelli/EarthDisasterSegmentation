
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from model.vit_layers import *
from model.layers import *
from model.loss import *
import os
import datetime

from utils.utils import instantiate_from_config





class TransUNet_AutoEncoder(tf.keras.layers.Layer):

    """
        U-Net AutoEncoder:

        All blocks are resblock. 
        
        Between encoder and decoder , there is vit block.
        
        Paper Ref:
        TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
        Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou

    """
    def __init__(self,hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_num_transformer=unet_num_transformer
        self.unet_num_heads=unet_num_heads
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

        self.middle_block=VIT(filter=self.unet_hidden_sizes[-1],embed_dim=self.unet_hidden_sizes[-1], num_transformer=self.unet_num_transformer,num_heads=self.unet_num_heads)


        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)

            skips.append(x)

        x=self.middle_block(x)
        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):

       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states
    

class TransUNetRES_AutoEncoder(tf.keras.layers.Layer):

    """
        U-Net AutoEncoder:

        All blocks are resblock. 
        
        Between encoder and decoder , there is vit block.
        
        Paper Ref:
        TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
        Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou

    """
    def __init__(self,hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_num_transformer=unet_num_transformer
        self.unet_num_heads=unet_num_heads
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
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

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

        self.middle_block=VIT(filter=self.unet_hidden_sizes[-1],embed_dim=self.unet_hidden_sizes[-1], num_transformer=self.unet_num_transformer,num_heads=self.unet_num_heads)


        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)

            skips.append(x)

        x=self.middle_block(x)
        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):

       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states
    


    
class UNet_TransformerAttn_AutoEncoder(tf.keras.layers.Layer):
    """
        first,second,third concatanetation has no cross attention , and also cross attention embedding dimesion reduced 4x .bec of computer power
        U-Net Transformer: Self and Cross Attention for Medical Image Segmentation
        ref :https://arxiv.org/pdf/2103.06104.pdf
    """
    def __init__(self,hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_num_transformer=unet_num_transformer
        self.unet_num_heads=unet_num_heads
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate
    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        self.vit_connection_blocks={}


        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)
                  if hidden_size in self.unet_hidden_sizes[3:]:
                     self.vit_connection_blocks[str(idx_x-1)]=MultiCrossAttention(embed_dim=hidden_size//4,out_embed_dim=hidden_size,num_heads=self.unet_num_heads)

        self.middle_blocks=[
            ConvBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
            ConvBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
            PositionalEncoding(),
            MultiHeadSelfAttention(embed_dim=self.unet_hidden_sizes[-1],num_heads=self.unet_num_heads)
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')



    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)

            skips.append(x)

        for idx,block in enumerate(self.middle_blocks):
            x=block(x)

        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):

       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  if ((len(self.decoder_blocks)-1)-idx) in list(self.vit_connection_blocks.keys()):
                        vitcross_block=self.vit_connection_blocks[str((len(self.decoder_blocks)-1)-idx)]
                        x,skip=vitcross_block([x,skips[(len(self.decoder_blocks)-1)-idx]])
                  else:
                        skip=skips[(len(self.decoder_blocks)-1)-idx]
                  x = tf.concat([x, skip], axis=-1)
                      
                  

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states
    
class UNet_Attention_AutoEncoder(tf.keras.layers.Layer):
    """
        Attention U-Net:
        Learning Where to Look for the Pancreas
        ref :https://arxiv.org/pdf/1804.03999.pdf
    """
    def __init__(self,hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_num_transformer=unet_num_transformer
        self.unet_num_heads=unet_num_heads
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate
    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        self.attn_connection_blocks={}


        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)
                 # if hidden_size in self.unet_hidden_sizes[3:]:
                  self.attn_connection_blocks[str(idx_x-1)]=AttentionGate(filters=hidden_size//2)

        self.middle_blocks=[
            ConvBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
            ConvBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')



    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)

            skips.append(x)

        for idx,block in enumerate(self.middle_blocks):
            x=block(x)

        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):

       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  if ((len(self.decoder_blocks)-1)-idx) in list(self.attn_connection_blocks.keys()):
                        attn_block=self.attn_connection_blocks[str((len(self.decoder_blocks)-1)-idx)]
                        skip=attn_block([x,skips[(len(self.decoder_blocks)-1)-idx]])
                  else:
                        skip=skips[(len(self.decoder_blocks)-1)-idx]
                  x = tf.concat([x, skip], axis=-1)
                      
                  

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states
    
class UNet_SE_AutoEncoder(tf.keras.layers.Layer):
    """
        USE-Net: incorporating Squeeze-and-Excitation blocks
        into U-Net for prostate zonal segmentation
        of multi-institutional MRI datasets
        ref :https://arxiv.org/pdf/1904.08254.pdf
    """
    def __init__(self,hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_num_transformer=unet_num_transformer
        self.unet_num_heads=unet_num_heads
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate
    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        self.se_connection_blocks={}
        self.final_se=SqueezeAndExcite2D(self.unet_hidden_sizes[0])


        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)
                 # if hidden_size in self.unet_hidden_sizes[3:]:
                  self.se_connection_blocks[str(idx_x-1)]=USENETBlock(filters=hidden_size//2)

        self.middle_blocks=[
            ConvBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
            ConvBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ConvBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)


        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')



    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)

            skips.append(x)

        for idx,block in enumerate(self.middle_blocks):
            x=block(x)

        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):

       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  if ((len(self.decoder_blocks)-1)-idx) in list(self.se_connection_blocks.keys()):
                        se_block=self.se_connection_blocks[str((len(self.decoder_blocks)-1)-idx)]
                        skip=se_block([x,skips[(len(self.decoder_blocks)-1)-idx]])
                  else:
                        skip=skips[(len(self.decoder_blocks)-1)-idx]
                  x = tf.concat([x, skip], axis=-1)
                      
                  

        x=tf.nn.relu(self.norm(x))
        x=self.final_se(x)
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states
         
    
class UNet_SEResNextBlock_AutoEncoder(tf.keras.layers.Layer):

    """
        U-Net AutoEncoder:

        Encoder first and second block is consists of convblock . They are like steem.
        The last four blocks of encoder are SE ResNeXt block.

        SE ResNeXt is a variant of a ResNext that employs squeeze-and-excitation blocks to enable the network to perform dynamic channel-wise feature recalibration.
        Paper Ref:
        For SE Original paper: 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
        FOR ResNeXt Original paper : Xie, Saining, et al. "Aggregated residual transformations for deep neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.


    """
    def __init__(self, hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks=unet_num_res_blocks
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate
    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                  idx_x=idx_x+1

                  if hidden_size in self.unet_hidden_sizes[:2]:
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                  else:
                    x = SEResBlock(hidden_size,cardinality=8,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    
                  self.encoder_blocks.append(x)
                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)
       

        self.middle_blocks=[SEResBlock(self.unet_hidden_sizes[-1],cardinality=8,norm=self.norm,drop_path_rate=self.drop_path_rate),
                            SEResBlock(self.unet_hidden_sizes[-1],cardinality=8,norm=self.norm,drop_path_rate=self.drop_path_rate)
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)

                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)

            skips.append(x)
        for block in self.middle_blocks:
            x=block(x)
        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):

       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states
    





class UNet_ResBlock_AutoEncoder(tf.keras.layers.Layer):
    """
    Basic U-Net with ResBlocks
    """
    def __init__(self, hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks=unet_num_res_blocks
        self.unet_hidden_sizes=[int(x) for x in hidden_sizes]
        self.unet_hidden_sizes.insert(0,hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,hidden_sizes[0]//2)
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate
    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
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

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

       

        self.middle_blocks=[ResBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate),
                            ResBlock(self.unet_hidden_sizes[-1],norm=self.norm,drop_path_rate=self.drop_path_rate)
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                hidden_states.append(x)
            x=block(x)
            skips.append(x)
        for block in self.middle_blocks:
            x=block(x)
        hidden_states.append(x)

        for idx,block in enumerate(self.decoder_blocks):
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states





class UNetModel(tf.keras.Model):
    def __init__(self, config,checkpoint_path,
                 special_checkpoint=None,

                 ):
        super(UNetModel,self).__init__()
        self.config=config
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        self.shape_input=config.input_shape
        self.use_ema=config.input_shape
        self.ema_momentum=config.ema_momentum
        self.gradient_clip_value=config.gradient_clip_value
        self.network=self.build_unet()
        self.loss_1_tracker = tf.keras.metrics.Mean(name="DiceLoss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="CrossEntropyLoss")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="iou")
        self.f1_score_tracker=tf.keras.metrics.Mean(name="f1")
        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint
        self.threshold_metric=config.threshold_metric
        self.loss_weights=config.loss_weights
  


    def build_unet(self,):
        input_image = tf.keras.Input(shape=self.shape_input)
        unet_layer=instantiate_from_config(self.config.unet)
        local_map,hidden_states=unet_layer(input_image)
        model = tf.keras.Model(inputs=input_image, outputs=local_map)
        return model

    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-05,)
        self.loss_1=DiceLoss()
        self.loss_2=tf.keras.losses.BinaryCrossentropy()
        self.iou_score=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        
    @property
    def metrics(self):
        return [
            self.loss_1_tracker,
            self.loss_2_tracker,
            self.iou_score_tracker,
            self.f1_score_tracker,
        ]
        

    def load(self,usage="train",return_epoch_number=True):
          self.checkpoint = tf.train.Checkpoint(
                                                network=self.network,
                                                optimizer=self.optimizer,
                                                step=tf.Variable(0),
                                                epoch=tf.Variable(0)
                                        )
      
          try:
            latest_checkpoint=tf.train.latest_checkpoint(self.checkpoint_dir)
          except:
            latest_checkpoint=None
            pass

          #tf.train.Checkpoint.restore(...).expect_partial() ignore errors
          if latest_checkpoint!=None and self.special_checkpoint==None:
              if usage=="eval":
                  self.checkpoint.restore(latest_checkpoint).expect_partial()
              else:
                  self.checkpoint.restore(latest_checkpoint)
              tf.print("-----------Restoring from {}-----------".format(latest_checkpoint))
              tf.print("-----------Step {}-----------".format(int(self.checkpoint.step)))

          elif self.special_checkpoint!=None:
              if usage=="eval":
                  self.checkpoint.restore(self.special_checkpoint).expect_partial()
              else:
                  self.checkpoint.restore(self.special_checkpoint)
              tf.print("-----------Restoring from {}-----------".format(
                  self.special_checkpoint))
              tf.print("-----------Step {}-----------".format(int(self.checkpoint.step)))

          else:

            tf.print("-----------Initializing from scratch-----------")
          
          if return_epoch_number==True:
            return int(self.checkpoint.epoch)
    
         


    def train_step(self, inputs):
        # 1. Get the batch size
        self.checkpoint.step.assign_add(1)
        tf.summary.scalar('steps', data=self.checkpoint.step, step=self.checkpoint.step)

        (x),(local_map)=inputs


        with tf.GradientTape() as tape:

            y_local = self.network(x, training=True)

            loss_1=self.loss_1(local_map,y_local)*self.loss_weights[0]
            loss_2=self.loss_2(local_map,y_local)*self.loss_weights[1]

            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(local_map,y_local)
        f1_score=(2*iou_score)/(1+iou_score)


        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.f1_score_tracker.update_state(f1_score)

        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x),(local_map)=inputs

        y_local = self.network(x, training=False)

       
        loss_1=self.loss_1(local_map,y_local)*self.loss_weights[0]
        loss_2=self.loss_2(local_map,y_local)*self.loss_weights[1]


        iou_score=self.iou_score(local_map,y_local)
        f1_score=(2*iou_score)/(1+iou_score)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.f1_score_tracker.update_state(f1_score)
        results = {m.name: m.result() for m in self.metrics}
        return results
