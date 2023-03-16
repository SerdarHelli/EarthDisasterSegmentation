#ref Huggingg Face Segformer

import tensorflow as tf
import math
from typing import Optional
from tensorflow import keras
from tensorflow.keras import layers
import os
from model.loss import *
import os
import datetime
from utils.utils import instantiate_from_config
from model.unet import *
import tensorflow.keras.backend as K



def gelu(x):
    tanh_res = tf.keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))
    return 0.5 * x * (1 + tanh_res)


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)

class GEGLU(keras.layers.Layer):
    def __init__(self, dim_out):
        super().__init__()
        self.proj = layers.Conv2D(dim_out*2,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.dim_out = dim_out

    def call(self, x):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
        return x * gelu(gate)
    
class ReScaler(keras.layers.Layer):
  def __init__(self, orig_shape):
    super().__init__()
    self.orig_shape=orig_shape
    
  def call(self, inputs):
      inputs=tf.image.resize(inputs,size=(self.orig_shape[1:3]))
      return inputs

class TFSegformerDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

class CrossAttentionBlock(layers.Layer):
    """Applies cross-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)
    
        self.query = layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.key = layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.value = layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.proj =layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')

    def build(self,input_shape):

        self.norm = layers.LayerNormalization()
        
    def call(self, inputs):
        inputs, context = inputs
        context = inputs if context is None else context

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(context)
        v = self.value(context)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj
    

class AttentionBlock(tf.keras.layers.Layer):
    """Applies self-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.query = tf.keras.layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.key = tf.keras.layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.value = tf.keras.layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.proj =tf.keras.layers.Conv2D(units,kernel_size=1,padding="same", kernel_initializer='he_normal')

    def build(self,input_shape):
        self.norm = layers.LayerNormalization()


        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim,):
        super().__init__()
        self.attn1 = AttentionBlock(dim)
        self.attn2 = CrossAttentionBlock(dim)
        self.geglu = GEGLU(dim * 4)
        self.dense  =layers.Conv2D(dim,kernel_size=1,padding="same", kernel_initializer='he_normal')

    def call(self, inputs):
        x, context = inputs
        x = self.attn1(x) + x
        x = self.attn2([x, context]) + x
        return self.dense(self.geglu(x)) + x


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = layers.LayerNormalization()
        self.proj_x =  tf.keras.layers.Conv2D(channels,kernel_size=1,padding="same", kernel_initializer='he_normal')

        self.proj_in =  tf.keras.layers.Conv2D(channels,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.transformer_blocks = [BasicTransformerBlock(channels)]
        self.proj_out = tf.keras.layers.Conv2D(channels,kernel_size=1,padding="same", kernel_initializer='he_normal')
        self.proj_out_final = tf.keras.layers.Conv2D(channels*4,kernel_size=1,padding="same", kernel_initializer='he_normal')

    def build(self,input_shape):
        shape=input_shape[0]
        self.scaler=ReScaler(orig_shape=shape)


    def call(self, inputs):
        x, context = inputs
        context = x if context is None else context
        if context is not None:
          context=self.scaler(context)
        x_in = self.proj_x(x)
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block([x, context])
        x=self.proj_out(x) + x_in

        return self.proj_out_final(x)
    

def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
        )(x)
        return x

    return apply

    
class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2DTranspose(self.filters , kernel_size=5, padding="same", strides=(2,2), kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
        self.act=tf.keras.layers.Activation("relu")

    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        x=self.norm1(x)
        x=self.act(x)
        return x

def getNorm(norm_str,eps=1e-6):
    x=None
    if norm_str=="batchnorm":
        x= tf.keras.layers.BatchNormalization(epsilon=eps)
    if norm_str=="layernorm":
        x= tf.keras.layers.LayerNormalization(epsilon=eps)
    if not x:
        raise("Invalid Normalization ")
    return x









class UnetSpatial_AutoEncoder(tf.keras.layers.Layer):

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

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(5, kernel_size=1,padding="same",name="classification_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor,context_tensor:tf.Tensor,hidden_sizes):
        x=self.conv_first(input_tensor)
        skips=[x]
        curr=0
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                x = tf.concat([x, hidden_sizes[curr]], axis=-1)
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
    

def build_uspatialcondition_model(
    input_shape,
    context_shape,
    widths,
    num_attention_heads,
    sr_ratios,
    has_attention,
    strides,
    patch_sizes,
    num_res_blocks=2,
    activation_fn=keras.activations.swish,
    first_conv_channels=16,
):
    
    hiddens_pre3= tf.keras.Input(shape=(16,16,widths[-1]),name="hiddenspre0")
    hiddens_pre2= tf.keras.Input(shape=(32,32,widths[-2]),name="hiddenspre1")
    hiddens_pre1= tf.keras.Input(shape=(64,64,widths[-3]),name="hiddenspre2")
    hiddens_pre0= tf.keras.Input(shape=(128,128,widths[-4]),name="hiddenspre3")
    hiddenspre=[hiddens_pre0,hiddens_pre1,hiddens_pre2,hiddens_pre3]

    hiddens_post3= tf.keras.Input(shape=(16,16,widths[-1]),name="hiddenspost0")
    hiddens_post2= tf.keras.Input(shape=(32,32,widths[-2]),name="hiddenspost1")
    hiddens_post1= tf.keras.Input(shape=(64,64,widths[-3]),name="hiddenspost2")
    hiddens_post0= tf.keras.Input(shape=(128,128,widths[-4]),name="hiddenspost3")
    hiddenspost=[hiddens_post0,hiddens_post1,hiddens_post2,hiddens_post3]

    pre_mask = layers.Input(
        shape=context_shape, name="pre_mask"
    )

    hiddens=[]
    for idx,(hidden_pre, hidden_post) in enumerate(zip(hiddenspre, hiddenspost)):
        x= tf.keras.layers.Concatenate()([hidden_pre,hidden_post])
        for _ in range(num_res_blocks):
            x=ResBlock(widths[-idx])(x)
        
        x=SpatialTransformer(widths[-idx]//4)([x,pre_mask])
        x=ReScaler(orig_shape=[1,128,128,widths[-4]])(x)
        hiddens.append(x)

    x= tf.keras.layers.Concatenate()(hiddens)
    for _ in range(num_res_blocks):
            x=ResBlock(widths[1])(x)

    x = UpSample(widths[1],norm="batchnorm")(x)

    for _ in range(num_res_blocks):
            x=ResBlock(widths[0])(x)

    x = UpSample(widths[0],norm="batchnorm")(x)
    x = tf.keras.layers.Conv2D(widths[0], 3, padding="same", kernel_initializer = 'he_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation("relu")

    x = tf.keras.layers.Conv2D(5,1, padding="same", )
    output=tf.keras.layers.Activation("softmax")

    return keras.Model([pre_mask,[hiddens_pre0,hiddens_pre1,hiddens_pre2,hiddens_pre3],[hiddens_post0,hiddens_post1,hiddens_post2,hiddens_post3]], output, name="uspatial_net")

def build_usegformernet_model(
    input_shape,
    context_shape,
    widths,
    num_attention_heads,
    sr_ratios,
    has_attention,
    strides,
    patch_sizes,
    num_res_blocks=2,
    activation_fn=keras.activations.swish,
    first_conv_channels=16,
):


    image_input = layers.Input(
        shape=input_shape, name="image_post"
    )
    image_input_pre= layers.Input(
        shape=context_shape, name="image_pre"
    )

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=3,
        padding="same",
    )(image_input)

    context = layers.Conv2D(
        first_conv_channels,
        kernel_size=1,
        padding="same",
    )(image_input_pre)


    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ConvBlock(
                widths[i]
            )(x)
            if has_attention[i]:

                  x=TFSegformerLayer(
                        hidden_size=widths[i],
                        num_attention_heads=num_attention_heads[i],
                        drop_path=0.1,
                        patch_size=patch_sizes[i],
                        stride=strides[i],
                        sequence_reduction_ratio=sr_ratios[i],
                        mlp_ratio=1,
                        attention_probs_dropout_prob=0.1,
                    )(x,context)


            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ConvBlock(widths[-1] )(
        x, 
    )

    x =   TFSegformerLayer(
                        hidden_size=widths[-1],
                        num_attention_heads=num_attention_heads[i],
                        drop_path=0.1,
                        patch_size=patch_sizes[-1],
                        stride=strides[-1],
                        sequence_reduction_ratio=sr_ratios[i],
                        mlp_ratio=1,
                        attention_probs_dropout_prob=0.1,
                    )(x,context)


    x = ConvBlock(widths[-1])(
        x,
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ConvBlock(
                widths[i]
            )(x)
            if has_attention[i]:
                  x=TFSegformerLayer(
                        hidden_size=widths[i],
                        num_attention_heads=num_attention_heads[i],
                        drop_path=0.1,
                        patch_size=patch_sizes[i],
                        stride=strides[i],
                        sequence_reduction_ratio=sr_ratios[i],
                        mlp_ratio=1,
                        attention_probs_dropout_prob=0.1,
                    )(x,context)


        if i != 0:
            x = UpSample(widths[i])(x)

    # End block
    x=getNorm("batchnorm")(x)
    x = activation_fn(x)
    x = layers.Conv2D(5, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("softmax")(x)

    return keras.Model([image_input, image_input_pre], x, name="usegformer_net")



class USegFormer(tf.keras.Model):
    def __init__(self, config,checkpoint_path,unet_config,unet_checkpoint_path,
                 special_checkpoint=None,
                 ):
        super(USegFormer,self).__init__()
        self.config=config
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        self.shape_input=config.input_shape
        self.use_ema=config.input_shape
        self.ema_momentum=config.ema_momentum
        self.gradient_clip_value=config.gradient_clip_value
        #self.unet_layer=None
        self.unet_layer=self.load_unetmodel(unet_config,unet_checkpoint_path)
        #self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=instantiate_from_config(config.usegformer)
        self.threshold_metric=config.threshold_metric
        self.loss_weights=config.loss_weights
        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="Crossentropy_Loss")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="Meaniou")
        self.f1_total_tracker=tf.keras.metrics.Mean(name="f1_total")
        self.f1_nodamage_tracker     =    tf.keras.metrics.Mean(name="f1_nodamage")
        self.f1_minordamage_tracker  = tf.keras.metrics.Mean(name="f1_minordamage")
        self.f1_majordamage_tracker  = tf.keras.metrics.Mean(name="f1_majordamage")
        self.f1_destroyed_tracker    = tf.keras.metrics.Mean(name="f1_destroyed")
        self.f1_unclassified_tracker = tf.keras.metrics.Mean(name="f1_unclassified")
        self.iou_score2_tracker=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1],name="iou")
        self.class_weights=config.weights


        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint
    
    def load_unetmodel(self,unet_config,unet_checkpoint_path):
        unet_model=UNetModel(unet_config,checkpoint_path=unet_checkpoint_path)
        unet_model.compile()
        print("Loading Unet Model")
        unet_model.load(usage="eval")
        layer_names=[layer.name for layer in unet_model.network.layers]
        unet_layer_trained=unet_model.network.get_layer(layer_names[-1])

        input_image = tf.keras.Input(shape=self.shape_input)
        unet_layer=instantiate_from_config(unet_config.unet)
        local_map,hidden_states=unet_layer(input_image)
        model = tf.keras.Model(inputs=input_image, outputs=[local_map,hidden_states])
        unet_layer.set_weights(unet_layer_trained.get_weights())
        return model


    # def build_usegformer(self,):
    #     input_pre = tf.keras.Input(shape=self.shape_input,name="pre_image")
    #     input_post= tf.keras.Input(shape=self.shape_input,name="post_image")
        
    #     local_map,hidden_states=self.unet_layer(input_pre)
    #     concatted=tf.keras.layers.Concatenate()([input_post,local_map])
    #     output=self.segformer_layer(concatted,hidden_states)
    #     model = tf.keras.Model(inputs=[input_pre,input_post], outputs=[output])
    #     return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,clipnorm=self.gradient_clip_value*2,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-04,)
        self.loss_1=DiceFocalLoss()
        self.loss_2=WeightedCrossentropy(weights = {0: 1,1: 1})
        self.loss_2_weighted=WeightedCrossentropy(weights = {0: 0.1,1: 2.})

        self.iou_score=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score1=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score2=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score3=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score4=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score5=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])

    @property
    def metrics(self):
        return [
            self.loss_1_tracker,
            self.loss_2_tracker,
            self.iou_score_tracker,
            self.f1_total_tracker,
            self.f1_nodamage_tracker,    
            self.f1_minordamage_tracker ,
            self.f1_majordamage_tracker ,
            self.f1_destroyed_tracker   ,
            self.f1_unclassified_tracker,
            self.iou_score2_tracker,
        ]
    


    def dice_classes_score(self,y_true, y_pred):
        dices={}
        d1=self.iou_score1(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d2=self.iou_score2(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d3=self.iou_score3(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d4=self.iou_score4(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d5=self.iou_score5(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        dices["nodamage"]=(2*d1)/(1+d1)
        dices["minordamage"]=(2*d2)/(1+d2)
        dices["majordamage"]=(2*d3)/(1+d3)
        dices["destroyed"]=(2*d4)/(1+d4)
        dices["unclassified"]=(2*d5)/(1+d5)
        dices["total_dice"]= (dices["nodamage"]+dices["minordamage"]+dices["majordamage"]+dices["unclassified"])/5
        return dices
    
    def loss1_full_compute(self,y_true, y_pred,weights=None):
        d1=self.loss_1(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d2=self.loss_1(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d3=self.loss_1(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d4=self.loss_1(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d5=self.loss_1(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))
        if weights:
            d1=d1*weights[0]
            d2=d2*weights[1]
            d3=d3*weights[2]
            d4=d4*weights[3]
            d5=d5*weights[4]
        loss=(d1+d2+d3+d4+d5)/5
        return loss

    def loss2_full_compute(self,y_true, y_pred,weights=None):
        d1=self.loss_2(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d2=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d3=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d4=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d5=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))
        if weights:
            d1=d1*weights[0]
            d2=d2*weights[1]
            d3=d3*weights[2]
            d4=d4*weights[3]
            d5=d5*weights[4]
        loss=(d1+d2+d3+d4+d5)/5
        return loss
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

        (x_pre,x_post),(local_map,multilabel_map)=inputs


        with tf.GradientTape() as tape:
            y_local,hiddens=self.unet_layer(x_pre,training=False)
            xx_pre=tf.concat([x_pre,y_local],axis=-1)

            y_multilabel_resized = self.network([x_post,xx_pre,hiddens], training=True)
            #upsample_resolution = tf.shape(multilabel_map)
     
            #y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

            loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
            loss_2=self.loss2_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[1]
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(K.flatten(multilabel_map[:,:,:,:4]),K.flatten(y_multilabel_resized[:,:,:,:4]))
        total_dice=(2*iou_score)/(1+iou_score)
        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.f1_total_tracker.update_state(total_dice)   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_unclassified_tracker.update_state(dices["unclassified"])
        self.iou_score2_tracker.update_state(K.flatten(multilabel_map[:,:,:,:4]),K.flatten(y_multilabel_resized[:,:,:,:4]))

        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_local,hiddens=self.unet_layer(x_pre,training=False)
        xx_pre=tf.concat([x_pre,y_local],axis=-1)
        y_multilabel_resized = self.network([x_post,xx_pre,hiddens], training=True)
        #upsample_resolution = tf.shape(multilabel_map)
  
        #y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

    

        loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
        loss_2=self.loss2_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[1]

        iou_score=self.iou_score(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
        total_dice=(2*iou_score)/(1+iou_score)
        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        self.f1_total_tracker.update_state(total_dice)   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_unclassified_tracker.update_state(dices["unclassified"])
        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.iou_score2_tracker.update_state(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))

        results = {m.name: m.result() for m in self.metrics}
        return results