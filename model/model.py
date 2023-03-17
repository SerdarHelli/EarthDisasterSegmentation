
import tensorflow as tf
from model.segformer import TFSegformerForSemanticSegmentation
from model.unet import *
import tensorflow.keras.backend as K
from model.loss import *
import os
import datetime
from utils.utils import instantiate_from_config



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
        self.loss_2_tracker = tf.keras.metrics.Mean(name="Crossentropy_Loss")
        self.f1_total_tracker=tf.keras.metrics.Mean(name="f1_total")
        self.f1_nodamage_tracker     =    tf.keras.metrics.Mean(name="f1_nodamage")
        self.f1_minordamage_tracker  = tf.keras.metrics.Mean(name="f1_minordamage")
        self.f1_majordamage_tracker  = tf.keras.metrics.Mean(name="f1_majordamage")
        self.f1_destroyed_tracker    = tf.keras.metrics.Mean(name="f1_destroyed")
        self.f1_background_tracker = tf.keras.metrics.Mean(name="f1_background")
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

        self.unet_layer=unet_model.network.get_layer(layer_names[-1])
        del unet_model

    def build_usegformer(self,):
        input_pre = tf.keras.Input(shape=self.shape_input,name="pre_image")
        input_post= tf.keras.Input(shape=self.shape_input,name="post_image")
        
        local_map,hidden_states=self.unet_layer(input_pre)
        concatted=tf.keras.layers.Concatenate()([input_post,local_map])
        output=self.segformer_layer(concatted,hidden_states)
        model = tf.keras.Model(inputs=[input_pre,input_post], outputs=[output])
        return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,clipnorm=self.gradient_clip_value*2,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-04,)
        self.loss_1=DiceFocalLoss()
        self.loss_2=tf.keras.losses.SparseCategoricalCrossentropy()

    @property
    def metrics(self):
        return [
            self.loss_1_tracker,
            self.loss_2_tracker,
            self.f1_total_tracker,
            self.f1_nodamage_tracker,    
            self.f1_minordamage_tracker ,
            self.f1_majordamage_tracker ,
            self.f1_destroyed_tracker   ,
            self.f1_background_tracker,
        ]
    


    def compute_tp_fn_fp(self,y_true, y_pred, c=1) :
        """
        Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)
        Args:
            y_pred (np.ndarray): prediction
            y_true (np.ndarray): target
            c (int): positive class
        """
        dices=[]
        for i in range(y_true.shape[0]):
            targ=y_true[i,:,:,:]
            pred=y_pred[i,:,:,:]

            pred=np.float32((y_pred>self.threshold_metric)*1)


            TP = np.logical_and(pred == c, targ == c).sum()
            FN = np.logical_and(pred != c, targ == c).sum()
            FP = np.logical_and(pred == c, targ != c).sum()
            
            R=np.float32((TP+1e-6)/(TP+FN+1e-6))
            P=np.float32((TP+1e-6)/(TP+FP+1e-6))
            dices.append(np.float32((2*P*R)/(P+R)))
                         
        dice=np.mean(np.asarray(dices))
        return np.float32(dice)
    
    def get_dice(self,y_true, y_pred):

        dice = tf.numpy_function(self.compute_tp_fn_fp, [y_true, y_pred], tf.float32)
        return dice

    def dice_classes_score(self,y_true, y_pred):
        y_true=tf.cast(y_true,dtype=tf.int32)
        y_true=tf.one_hot(y_true, 5)
        dices={}
        d0=self.get_dice(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d1=self.get_dice(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d2=self.get_dice(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d3=self.get_dice(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d4=self.get_dice(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        dices["nodamage"]=(2*d1)/(1+d1)
        dices["minordamage"]=(2*d2)/(1+d2)
        dices["majordamage"]=(2*d3)/(1+d3)
        dices["destroyed"]=(2*d4)/(1+d4)
        dices["background"]=(2*d0)/(1+d0)
        dices["total_dice"]= 4/((dices["nodamage"]+1e-6)**-1+(dices["minordamage"]+1e-6)**-1+(dices["majordamage"]+1e-6)**-1+(dices["destroyed"]+1e-6)**-1)
        return dices
    
    def loss1_full_compute(self,y_true, y_pred,weights=None):
        y_true=tf.cast(y_true,dtype=tf.int32)
        y_true=tf.one_hot(y_true, 5)
        d0=self.loss_1(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d1=self.loss_1(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d2=self.loss_1(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d3=self.loss_1(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d4=self.loss_1(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))
        if weights:
            d0=d0*weights[0]
            d1=d1*weights[1]
            d2=d2*weights[2]
            d3=d3*weights[3]
            d4=d4*weights[4]
        loss=(d1+d2+d3+d4+d0)/5
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

            y_multilabel = self.network([x_pre,x_post], training=True)
            upsample_resolution = tf.shape(multilabel_map)
     
            y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

            loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
            loss_2=self.loss_2(multilabel_map,y_multilabel_resized,)*self.loss_weights[1]
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
        total_dice=(2*iou_score)/(1+iou_score)
        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        self.f1_total_tracker.update_state(dices["total_dice"])   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_background_tracker.update_state(dices["background"])
        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)

        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_multilabel = self.network([x_pre,x_post], training=False)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

    

        loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
        loss_2=self.loss_2(multilabel_map,y_multilabel_resized,)*self.loss_weights[1]


        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        self.f1_total_tracker.update_state(dices["total_dice"])   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_background_tracker.update_state(dices["background"])
        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)

        results = {m.name: m.result() for m in self.metrics}
        return results





class USegFormerSeperated(tf.keras.Model):
    def __init__(self, config,checkpoint_path,unet_config,unet_checkpoint_path,
                 special_checkpoint=None,
                 ):
        super(USegFormerSeperated,self).__init__()
        self.config=config
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        self.shape_input=config.input_shape
        self.use_ema=config.input_shape
        self.ema_momentum=config.ema_momentum
        self.gradient_clip_value=config.gradient_clip_value
        self.unet_layer=self.load_unetmodel(unet_config,unet_checkpoint_path)
        self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=self.build_usegformer()
        self.threshold_metric=config.threshold_metric
        self.loss_weights=config.loss_weights

        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="Crossentropy_Loss")
        self.iou_score2_tracker=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1],name="iou")
        self.f1_total_tracker=tf.keras.metrics.Mean(name="f1_total")
        self.f1_nodamage_tracker     =    tf.keras.metrics.Mean(name="f1_nodamage")
        self.f1_minordamage_tracker  = tf.keras.metrics.Mean(name="f1_minordamage")
        self.f1_majordamage_tracker  = tf.keras.metrics.Mean(name="f1_majordamage")
        self.f1_destroyed_tracker    = tf.keras.metrics.Mean(name="f1_destroyed")
        self.f1_background_tracker = tf.keras.metrics.Mean(name="f1_background")
        self.f1_local_tracker    =    tf.keras.metrics.Mean(name="f1_local")
        self.iou_score_tracker=tf.keras.metrics.Mean(name="Meaniou")

        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint
        self.class_weights=config.weights
        
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

    def build_usegformer(self,):
        post_image = tf.keras.Input(shape=self.shape_input,name="post_image")
        pre_target = tf.keras.Input(shape=self.unet_layer.output[0].shape[1:],name="pre_image")

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
        x=tf.keras.layers.Concatenate()([post_image,pre_target])

        output=self.segformer_layer(x,hiddens)
        
        model = tf.keras.Model(inputs=[post_image,pre_target,[hiddens_0,hiddens_1,hiddens_2,hiddens_3]], outputs=[output])
        return model


    def compile(self,**kwargs):
        super().compile(**kwargs)
        self.optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=self.lr ,weight_decay=self.weight_decay,clipvalue=self.gradient_clip_value,clipnorm=self.gradient_clip_value*2,
                                                              use_ema=self.use_ema,ema_momentum=self.ema_momentum,epsilon=1e-04,)
        self.loss_1=DiceLoss(weight=[ .4 , .4 , 2.4 , 1.2 ,.8])
        self.loss_2=tf.keras.losses.BinaryCrossentropy()
        self.iou_score=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score1=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score2=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score3=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score4=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score5=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])

        self.iou_score_local=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])


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
            self.f1_background_tracker,
            self.f1_local_tracker,
            self.iou_score2_tracker

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
        dices["background"]=(2*d5)/(1+d5)
        dices["total_dice"]= (dices["nodamage"]+dices["minordamage"]+dices["majordamage"]+dices["background"])/5
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
            loss=(d1+d2+d3+d4+d5)
            return loss
        loss=(d1+d2+d3+d4+d5)/5
        return loss

    def loss2_full_compute(self,y_true, y_pred,weights=None):
        d1=self.loss_2(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d2=self.loss_2(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d3=self.loss_2(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d4=self.loss_2(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d5=self.loss_2(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))
        if weights:
            d1=d1*weights[0]
            d2=d2*weights[1]
            d3=d3*weights[2]
            d4=d4*weights[3]
            d5=d5*weights[4]
            loss=(d1+d2+d3+d4+d5)
            return loss      

        loss=(d1+d2+d3+d4+d5)/5
        return loss
    def train_step(self, inputs):
        # 1. Get the batch size
        self.checkpoint.step.assign_add(1)
        tf.summary.scalar('steps', data=self.checkpoint.step, step=self.checkpoint.step)

        (x_pre,x_post),(local_map,multilabel_map)=inputs


        with tf.GradientTape() as tape:
            y_local,hiddens=self.unet_layer(x_pre,training=False)

            y_multilabel = self.network([x_post,y_local,hiddens], training=True)
            upsample_resolution = tf.shape(multilabel_map)
     
            y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

            loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
            loss_2=self.loss2_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[1]
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        iou_score=self.iou_score(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
        total_dice=(2*iou_score)/(1+iou_score)

        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        local_iou=self.iou_score_local(K.flatten(local_map),K.flatten(y_local))
        local_f1=(2*local_iou)/(1+local_iou)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.f1_total_tracker.update_state(total_dice)   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_background_tracker.update_state(dices["background"])
        self.f1_local_tracker.update_state(local_f1)
        self.iou_score2_tracker.update_state(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_local,hiddens=self.unet_layer(x_pre,training=False)
        y_multilabel = self.network([x_post,y_local,hiddens], training=False)
        upsample_resolution = tf.shape(multilabel_map)
  
        y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

        loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
        loss_2=self.loss2_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[1]

        iou_score=self.iou_score(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))
        total_dice=(2*iou_score)/(1+iou_score)
        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        local_iou=self.iou_score_local(K.flatten(local_map),K.flatten(y_local))
        local_f1=(2*local_iou)/(1+local_iou)

        self.f1_total_tracker.update_state(total_dice)   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_background_tracker.update_state(dices["background"])
        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.iou_score_tracker.update_state(iou_score)
        self.f1_local_tracker.update_state(local_f1)
        self.iou_score2_tracker.update_state(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))

        results = {m.name: m.result() for m in self.metrics}
        return results
