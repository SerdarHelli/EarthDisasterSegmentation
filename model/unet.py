
import tensorflow as tf
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from model.vit_layers import VIT
from model.loss import *
import os
import datetime



class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.learned_skip = False
        self.batch_norm1 = tf.keras.layers.BatchNormalization( )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
            self.batch_norm3 = tf.keras.layers.BatchNormalization()
        

    def call(self, input_tensor: tf.Tensor):
        x = self.batch_norm1(input_tensor)
        x = self.conv_1(tf.nn.relu(x))
        x = self.batch_norm2(x)
        x = self.conv_2(tf.nn.relu(x))
        skip = (
            self.conv_3(tf.nn.relu(self.batch_norm3(input_tensor)))
            if self.learned_skip
            else input_tensor
        )
        output = skip + x

        return output




class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2DTranspose(self.filters , kernel_size=5, padding="same", strides=(2,2), kernel_initializer = 'he_normal')
    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        return x
    




class UTransNet_AutoEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.unet_hidden_sizes=[int(x) for x in config.hidden_sizes]
        self.unet_hidden_sizes.insert(0,config.hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,config.hidden_sizes[0]//2)

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.config.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.config.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ResBlock(hidden_size)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

       

        self.middle_blocks=[ResBlock(self.unet_hidden_sizes[-1]),
                            ResBlock(self.unet_hidden_sizes[-1])
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.config.unet_num_res_blocks):
                    x = ResBlock(hidden_size)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size)
                   self.decoder_blocks.append(x)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.middle_block=VIT(filter=self.unet_hidden_sizes[-1],embed_dim=self.unet_hidden_sizes[-1], num_transformer=self.config.unet_num_transformer,num_heads=self.config.unet_num_heads)
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

        x=tf.nn.relu(self.batch_norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states


class UNet_AutoEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.unet_hidden_sizes=[int(x) for x in config.hidden_sizes]
        self.unet_hidden_sizes.insert(0,config.hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,config.hidden_sizes[0]//2)

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.config.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.config.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ResBlock(hidden_size)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

       

        self.middle_blocks=[ResBlock(self.unet_hidden_sizes[-1]),
                            ResBlock(self.unet_hidden_sizes[-1])
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.config.unet_num_res_blocks):
                    x = ResBlock(hidden_size)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size)
                   self.decoder_blocks.append(x)

        self.batch_norm = tf.keras.layers.BatchNormalization()
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

        x=tf.nn.relu(self.batch_norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x,hidden_states




"""
from omegaconf import OmegaConf
USegformerConfig={
     "num_channels": 3, "num_encoder_blocks" : 4 ,"depths" : [2, 2, 2, 2] ,"sr_ratios" : [8, 4, 2, 1],"lr":1.5e-4,"weight_decay":1e-6,
      "hidden_sizes" : [32, 64, 160, 256], "patch_sizes" : [7, 3, 3, 3] ,"use_ema":True,"ema_momentum":0.9999,
      "strides" : [4, 2, 2, 2], "num_attention_heads" : [1, 2, 5, 8] ,"mlp_ratios" : [4, 4, 4, 4] ,  "attention_probs_dropout_prob" : 0.0,"output_hidden_states":False,"output_attentions":False,"gradient_clip_value" : 1,
      "classifier_dropout_prob" : 0.1 ,"use_return_dict":True, "layer_norm_eps" : 1e-06,"reshape_last_stage":True, "input_shape":[512,512,3],
      "decoder_hidden_size" : 256,"num_labels":5,"unet_num_res_blocks":2,'unet_num_heads':4,'unet_num_transformer':4,

}

conf = OmegaConf.structured(USegformerConfig)
"""

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
        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="FocalTversky_loss")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="iou")

        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint

  


    def build_unet(self,):
        input_image = tf.keras.Input(shape=self.shape_input)
        local_map,hidden_states=UNet_AutoEncoder(self.config)(input_image)
        model = tf.keras.Model(inputs=input_image, outputs=local_map)
        return model

    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-05,)
        self.loss_1=DiceLoss()
        self.loss_2=FocalTverskyLoss()


    @property
    def metrics(self):
        return [
            self.loss_1_tracker,
            self.loss_2_tracker,
            self.iou_score_tracker,
        ]
    

    def iou_score(self,y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou
    
    def dice_score(self,y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice= K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return dice


        


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

            loss_1=self.loss_1(local_map,y_local)
            loss_2=self.loss_2(local_map,y_local)

            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(local_map,y_local)


        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x),(local_map)=inputs

        y_local = self.network(x, training=False)

        loss_2=self.loss_2(local_map,y_local)
        loss_1=self.loss_1(local_map,y_local)
        iou_score=self.iou_score(local_map,y_local)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        results = {m.name: m.result() for m in self.metrics}
        return results
