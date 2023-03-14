
import tensorflow as tf
from model.segformer import TFSegformerForSemanticSegmentation
from model.unet import *
import tensorflow.keras.backend as K
from model.loss import *
import os
import datetime



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
        self.unet_layer=None
        self.load_unetmodel(unet_config,unet_checkpoint_path)
        self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=self.build_usegformer()
        self.threshold_metric=config.threshold_metric
        self.loss_weights=config.loss_weights
        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="WeightedCategoricalCrossentropy_Loss")
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
    
    def load_unetmodel(self,unet_config,unet_checkpoint_path):
        unet_model=UNetModel(unet_config,checkpoint_path=unet_checkpoint_path)
        unet_model.compile()
        print("Loading Unet Model")
        unet_model.load(usage="eval")
        layer_names=[layer.name for layer in unet_model.network.layers]

        self.unet_layer=unet_model.network.get_layer(layer_names[-1])
        self.unet_layer.trainable=False
        del unet_model

    def build_usegformer(self,):
        input_pre = tf.keras.Input(shape=self.shape_input,name="pre_image")
        input_post= tf.keras.Input(shape=self.shape_input,name="post_image")
        local_map,hidden_states=self.unet_layer(input_pre)
        self.unet_layer.trainable=False
        #x=SPADE(filters=self.shape_input[-1])(input_post,local_map)
        x=tf.keras.layers.Concatenate()([input_post,local_map])
        output=self.segformer_layer(x,hidden_states)
        model = tf.keras.Model(inputs=[input_pre,input_post], outputs=[output])
        return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,clipnorm=self.gradient_clip_value*2,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-04,)
        self.loss_1=DiceLoss(weight=[ .4 , .4 , 2.4 , 1.2 ,.8])
        self.loss_2=tf.keras.losses.BinaryCrossentropy()
        self.iou_score=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])


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
    


    
    def dice_classes_score(self,y_true, y_pred):
        dices={}
        d1=self.iou_score(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d2=self.iou_score(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d3=self.iou_score(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d4=self.iou_score(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d5=self.iou_score(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        dices["nodamage"]=(2*d1)/(1+d1)
        dices["minordamage"]=(2*d2)/(1+d2)
        dices["majordamage"]=(2*d3)/(1+d3)
        dices["destroyed"]=(2*d4)/(1+d4)
        dices["unclassified"]=(2*d5)/(1+d5)
        dices["total_dice"]= (dices["nodamage"]+dices["minordamage"]+dices["majordamage"]+dices["unclassified"])/5
        return dices





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

            loss_1=self.loss_1(multilabel_map,y_multilabel_resized)*self.loss_weights[0]
            loss_2=self.loss_2(multilabel_map,y_multilabel_resized)*self.loss_weights[1]
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
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


        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_multilabel = self.network([x_pre,x_post], training=False)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

    

        loss_1=self.loss_1(multilabel_map,y_multilabel_resized)*self.loss_weights[0]
        loss_2=self.loss_2(multilabel_map,y_multilabel_resized)*self.loss_weights[1]

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
        results = {m.name: m.result() for m in self.metrics}
        return results





class USegFormerSeperated(tf.keras.Model):
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
        self.unet_layer=None
        self.load_unetmodel(unet_config,unet_checkpoint_path)
        self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=self.build_usegformer()
        self.threshold_metric=config.threshold_metric
        self.loss_weights=config.loss_weights
        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="WeightedCategoricalCrossentropy_Loss")
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
    
    def load_unetmodel(self,unet_config,unet_checkpoint_path):
        unet_model=UNetModel(unet_config,checkpoint_path=unet_checkpoint_path)
        unet_model.compile()
        print("Loading Unet Model")
        unet_model.load(usage="eval")
        layer_names=[layer.name for layer in unet_model.network.layers]

        self.unet_layer=unet_model.network.get_layer(layer_names[-1])
        self.unet_layer.trainable=False
        del unet_model

    def build_usegformer(self,):
        post_image = tf.keras.Input(shape=self.shape_input,name="post_image")
        pre_target = tf.keras.Input(shape=self.shape_input,name="pre_image")

        shapes=[]
        for hiddens in self.unet_layer.output[1]:
            shapes.append(hiddens.shape[1:])
        if len(self.unet_layer.output[1])!=4:
            raise "Shape Error"

        hiddens_0= tf.keras.Input(shape=shapes[0],name="hiddens0")
        hiddens_1= tf.keras.Input(shape=shapes[1],name="hiddens1")
        hiddens_2= tf.keras.Input(shape=shapes[2],name="hiddens2")
        hiddens_3= tf.keras.Input(shape=shapes[3],name="hiddens3")
        hiddens=[hiddens_0,hiddens_1,hiddens_2,hiddens_3]
        concatted_post_image=tf.keras.layers.Concatenate()([post_image,pre_target])

        output=self.segformer_layer(concatted_post_image,hiddens)
        model = tf.keras.Model(inputs=[post_image,pre_target,[hiddens_0,hiddens_1,hiddens_2,hiddens_3]], outputs=[output])
        return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,clipnorm=self.gradient_clip_value*2,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-04,)
        self.loss_1=DiceLoss(weight=[ .4 , .4 , 2.4 , 1.2 ,.8])
        self.loss_2=tf.keras.losses.BinaryCrossentropy()
        self.iou_score=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])


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
  
    def dice_classes_score(self,y_true, y_pred):
        dices={}
        d1=self.iou_score(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d2=self.iou_score(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d3=self.iou_score(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d4=self.iou_score(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d5=self.iou_score(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        dices["nodamage"]=(2*d1)/(1+d1)
        dices["minordamage"]=(2*d2)/(1+d2)
        dices["majordamage"]=(2*d3)/(1+d3)
        dices["destroyed"]=(2*d4)/(1+d4)
        dices["unclassified"]=(2*d5)/(1+d5)
        dices["total_dice"]= (dices["nodamage"]+dices["minordamage"]+dices["majordamage"]+dices["unclassified"])/5
        return dices

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
            
            y_local,hiddens=self.unet_layer(x_pre)
            y_multilabel = self.network([x_post,y_local,hiddens], training=True)
            upsample_resolution = tf.shape(multilabel_map)
     
            y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

            loss_1=self.loss_1(multilabel_map,y_multilabel_resized)*self.loss_weights[0]
            loss_2=self.loss_2(multilabel_map,y_multilabel_resized)*self.loss_weights[1]
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
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


        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_local,hiddens=self.unet_layer(x_pre)
        y_multilabel = self.network([x_post,y_local,hiddens], training=True)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

        loss_1=self.loss_1(multilabel_map,y_multilabel_resized)*self.loss_weights[0]
        loss_2=self.loss_2(multilabel_map,y_multilabel_resized)*self.loss_weights[1]

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
        results = {m.name: m.result() for m in self.metrics}
        return results
