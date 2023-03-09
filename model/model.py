
import tensorflow as tf
from model.segformer import TFSegformerForSemanticSegmentation
from model.unet import *
from model.layers import SPADE
import numpy as np
import tensorflow.keras.backend as K
from model.loss import *
import os
import datetime



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
        self.unet_layer=None
        self.load_unetmodel(config,unet_checkpoint_path)
        self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=self.build_usegformer()
        self.threshold_value=0.1
        
        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="GeneralizedFocalTversky_Loss")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="iou")
        self.f1_total_tracker=tf.keras.metrics.Mean(name="f1_total")
        self.f1_nodamage_tracker     =    tf.keras.metrics.Mean(name="f1_nodamage")
        self.f1_minordamage_tracker  = tf.keras.metrics.Mean(name="f1_minordamage")
        self.f1_majordamage_tracker  = tf.keras.metrics.Mean(name="f1_majordamage")
        self.f1_destroyed_tracker    = tf.keras.metrics.Mean(name="f1_destroyed")
        self.f1_unclassified_tracker = tf.keras.metrics.Mean(name="f1_unclassified")


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
        self.unet_layer=unet_model.network.get_layer("u_net__auto_encoder" )
        self.unet_layer.trainable=False
        del unet_model

    def build_usegformer(self,):
        input_pre = tf.keras.Input(shape=self.shape_input,name="pre_image")
        input_post= tf.keras.Input(shape=self.shape_input,name="post_image")
        local_map,hidden_states=self.unet_layer(input_pre)
        self.unet_layer.trainable=False
        #x=SPADE(filters=self.shape_input[-1])(input_post,local_map)
        concatted = tf.keras.layers.Concatenate()([input_post, local_map])
        output=self.segformer_layer(concatted,hidden_states)
        model = tf.keras.Model(inputs=[input_pre,input_post], outputs=[output])
        return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,clipnorm=self.gradient_clip_value*2,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-04,)
        self.loss_1=DiceLoss(weight=[ .4 , .4 , 2.4 , 1.2 ,.8])
        self.loss_2=GeneralizedFocalTverskyLoss()
        self.iou_score=tf.keras.metrics.OneHotIoU(num_classes=5, target_class_id=[0,1,2,3,4])


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

        ]
    
    def recall_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self,y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_score(self,y_true, y_pred):
        y_pred = tf.cast(y_pred >= (self.threshold_metric), tf.float32)
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    
    def dice_classes_score(self,y_true, y_pred):
        dices={}
        dices["nodamage"]=self.f1_score(y_true[:,:,:,0],y_pred[:,:,:,0])
        dices["minordamage"]=self.f1_score(y_true[:,:,:,1],y_pred[:,:,:,1])
        dices["majordamage"]=self.f1_score(y_true[:,:,:,2],y_pred[:,:,:,2])
        dices["destroyed"]=self.f1_score(y_true[:,:,:,3],y_pred[:,:,:,3])
        dices["unclassified"]=self.f1_score(y_true[:,:,:,4],y_pred[:,:,:,4])
        total_dice=(dices["unclassified"]+  dices["destroyed"]+dices["majordamage"]+dices["minordamage"]+dices["nodamage"])/5
        return dices,total_dice





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

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        dices,total_dice=self.dice_classes_score(multilabel_map,y_multilabel_resized)
        self.f1_total_tracker.update_state(total_dice)   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_unclassified_tracker.update_state(dices["unclassified"])


        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_multilabel = self.network([x_pre,x_post], training=False)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

        
        loss_1=self.loss_1(multilabel_map,y_multilabel_resized)
        loss_2=self.loss_2(multilabel_map,y_multilabel_resized)

        iou_score=self.iou_score(multilabel_map,y_multilabel_resized)
        dices,total_dice=self.dice_classes_score(multilabel_map,y_multilabel_resized)
        self.f1_total_tracker.update_state(total_dice)   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_unclassified_tracker.update_state(dices["unclassified"])
        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        results = {m.name: m.result() for m in self.metrics}
        return results
