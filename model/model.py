
import tensorflow as tf
from model.segformer import TFSegformerForSemanticSegmentation
from model.unet import *
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
      "strides" : [4, 2, 2, 2], "num_attention_heads" : [1, 2, 5, 8] ,"mlp_ratios" : [4, 4, 4, 4] ,  "attention_probs_dropout_prob" : 0.0,"output_hidden_states":False,"output_attentions":False,"gradient_clip_value" : 1,
      "classifier_dropout_prob" : 0.1 ,"use_return_dict":True, "layer_norm_eps" : 1e-06,"reshape_last_stage":True, "input_shape":[512,512,3],
      "decoder_hidden_size" : 256,"num_labels":5,"unet_num_res_blocks":2,'unet_num_heads':4,'unet_num_transformer':4,"drop_path_rate":0.1
}
conf = OmegaConf.structured(USegformerConfig)
"""

class USegFormer(tf.keras.Model):
    def __init__(self, config,checkpoint_path,unet_checkpoint_path,
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

        self.load_unetmodel(config,unet_checkpoint_path)

        self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=self.build_usegformer()
        self.threshold_value=0.25
        self.loss_1_tracker = tf.keras.metrics.Mean(name="GenDice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="FocalTversky_loss")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="iou")
        self.challenge_score_tracker= tf.keras.metrics.Mean(name="challenge_score")

        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint
    
    def load_unetmodel(self,config,unet_checkpoint_path):
        unet_model=UNetModel(config,checkpoint_path=unet_checkpoint_path)
        unet_model.compile()
        print("Loading Unet Model")
        unet_model.load()
        self.unet_layer=self.unet_model.network.get_layer("u_net__auto_encoder" )
        del unet_model


    def build_usegformer(self,):
        input_pre = tf.keras.Input(shape=self.shape_input,name="pre_image")
        input_post= tf.keras.Input(shape=self.shape_input,name="post_image")
        local_map,hidden_states=self.unet_layer(input_pre)
        concatted = tf.keras.layers.Concatenate()([input_post, local_map])
        output=self.segformer_layer(concatted,hidden_states)
        model = tf.keras.Model(inputs=[input_pre,input_post], outputs=[output])
        return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-05,)
        self.loss_1=GeneralizedDice()
        self.loss_2=FocalTverskyLoss()


    @property
    def metrics(self):
        return [
            self.loss_1_tracker,
            self.loss_1_local_tracker,
            self.loss_2_tracker,
            self.loss_2_local_tracker,
            self.iou_score_local_tracker,
            self.iou_score_tracker,
            self.challenge_score_tracker,
        ]
    
    def make_threshold(self,multi_label,):
        array=(multi_label>float(self.threshold_value))*1
        return np.float32(np.expand_dims(array,axis=-1))

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


    def challange_score(self,y_true,y_pred):
        y_true_loc=K.sum(y_true,axis=[3])
        y_true_loc = tf.numpy_function(self.make_threshold, [y_true_loc], tf.float32)
        y_pred_loc=K.sum(y_pred,axis=[3])
        y_pred_loc = tf.numpy_function(self.make_threshold, [y_pred_loc], tf.float32)
        loc_dice=self.dice_score(y_true_loc,y_pred_loc)
        multi_dice=self.dice_score(y_true,y_pred)
        score=(loc_dice*0.3)+(multi_dice*0.7)
        return score
        


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

            y_multilabel = self.network([x_pre,x_post], training=True)
            upsample_resolution = tf.shape(multilabel_map)
     
            y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")



            loss_1=self.loss_1(multilabel_map,y_multilabel_resized)
            loss_2=self.loss_2(multilabel_map,y_multilabel_resized)
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(multilabel_map,y_multilabel_resized)
        challenge_score=self.challange_score(multilabel_map,y_multilabel_resized)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.challenge_score_tracker.update_state(challenge_score)
        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_multilabel = self.network([x_pre,x_post], training=False)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

        
        loss_2=self.loss_1(multilabel_map,y_multilabel_resized)
        loss_1=self.loss_2(multilabel_map,y_multilabel_resized)

        iou_score=self.iou_score(multilabel_map,y_multilabel_resized)
        challenge_score=self.challange_score(multilabel_map,y_multilabel_resized)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.challenge_score_tracker.update_state(challenge_score)
        results = {m.name: m.result() for m in self.metrics}
        return results
