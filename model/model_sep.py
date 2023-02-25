
#To-DO 
#Seperated Training ,Combination of Two models Unet and Segformer


import tensorflow as tf
from model.segformer import TFSegformerForSemanticSegmentation
from model.unet import UNet
import numpy as np
import tensorflow.keras.backend as K
from model.loss import *
import os
import datetime

"""
from omegaconf import OmegaConf
USegformerConfig={
     "num_channels": 3, "num_encoder_blocks" : 4 ,"depths" : [2, 2, 2, 2] ,"sr_ratios" : [8, 4, 2, 1],"lr":1.5e-4,"weight_decay":1e-6,
      "hidden_sizes" : [32, 64, 160, 256], "patch_sizes" : [7, 3, 3, 3] ,"use_ema":True,"ema_momentum":0.9999,
      "strides" : [4, 2, 2, 2], "num_attention_heads" : [1, 2, 5, 8] ,"mlp_ratios" : [4, 4, 4, 4] ,"hidden_act" : 'gelu', "hidden_dropout_prob" : 0.0,
      "attention_probs_dropout_prob" : 0.0,"output_hidden_states":False,"output_attentions":False,"gradient_clip_value" : 2,
      "classifier_dropout_prob" : 0.1 ,"initializer_range" : 0.02,"use_return_dict":True,"classifier_dropout_prob":0.0,
      "drop_path_rate" : 0.1, "layer_norm_eps" : 1e-06,"reshape_last_stage":True, "input_shape":[512,512,3],
      "decoder_hidden_size" : 256, "semantic_loss_ignore_index" : 255,"num_labels":5,"unet_num_res_blocks":3

}

conf = OmegaConf.structured(USegformerConfig)
"""

class USegFormer(tf.keras.Model):
    def __init__(self, config,checkpoint_path,
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
        self.unet_layer = UNet(config)
        self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=self.build_usegformer()

        self.combo_loss_tracker = tf.keras.metrics.Mean(name="Combo_loss")
        self.combo_local_loss_tracker = tf.keras.metrics.Mean(name="local_Combo_loss")
        self.focal_loss_tracker = tf.keras.metrics.Mean(name="FocalTversky_loss")
        self.focal_local_loss_tracker = tf.keras.metrics.Mean(name="local_FocalTversky_loss")
        self.iou_score_local_tracker = tf.keras.metrics.Mean(name="iou_local")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="iou")
        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint


    def build_segformer(self,):
        input_post_concatted= tf.keras.Input(shape=self.shape_input,name="post_concataned_image")

        local_map,hidden_states=self.unet_layer(input_pre)
        concatted = tf.keras.layers.Concatenate()([input_post, local_map])

        output=self.segformer_layer(concatted,hidden_states)

        model = tf.keras.Model(inputs=[input_pre,input_post], outputs=[local_map,output])
        return model

    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-05,)
        self.combo_loss=ComboLoss()
        self.focal_loss=FocalTverskyLoss()


    @property
    def metrics(self):
        return [
            self.combo_loss_tracker,
            self.combo_local_loss_tracker,
            self.focal_loss_tracker,
            self.focal_local_loss_tracker,
            self.iou_score_local_tracker,
            self.iou_score_tracker,

        ]

    def iou_score(self,y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

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

            y_local,y_multilabel = self.network([x_pre,x_post], training=True)
            upsample_resolution = tf.shape(multilabel_map)
     
            y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")



            loss_combo=self.combo_loss(multilabel_map,y_multilabel_resized)
            loss_focal=self.focal_loss(multilabel_map,y_multilabel_resized)
            loss_local_combo=self.combo_loss(local_map,y_local)
            loss_local_focal=self.focal_loss(local_map,y_local)
            loss=loss_combo+loss_focal+loss_local_combo+loss_local_focal

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(multilabel_map,y_multilabel_resized)
        iou_local_score=self.iou_score(local_map,y_local)


        self.combo_loss_tracker.update_state(loss_combo)
        self.combo_local_loss_tracker.update_state(loss_local_combo)
        self.focal_loss_tracker.update_state(loss_focal)
        self.focal_local_loss_tracker.update_state(loss_local_focal)
        self.iou_score_local_tracker.update_state(iou_local_score)
        self.iou_score_tracker.update_state(iou_score)
        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_local,y_multilabel = self.network([x_pre,x_post], training=False)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")


        loss_local_combo=self.combo_loss(local_map,y_local)
        loss_local_focal=self.focal_loss(local_map,y_local)
        
        loss_combo=self.combo_loss(multilabel_map,y_multilabel_resized)
        loss_focal=self.focal_loss(multilabel_map,y_multilabel_resized)

        iou_score=self.iou_score(multilabel_map,y_multilabel_resized)
        iou_local_score=self.iou_score(y_local,local_map)


        self.combo_loss_tracker.update_state(loss_combo)
        self.combo_local_loss_tracker.update_state(loss_local_combo)
        self.focal_loss_tracker.update_state(loss_focal)
        self.focal_local_loss_tracker.update_state(loss_local_focal)
        self.iou_score_local_tracker.update_state(iou_local_score)
        self.iou_score_tracker.update_state(iou_score)
        results = {m.name: m.result() for m in self.metrics}
        return results
