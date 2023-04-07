
from model.layers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from model.segformer_components import *
from model.loss import *



class UsegFormerClass(tf.keras.Model):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      
      self.loss_1= DiceFocalLoss()
      self.loss_2=tf.keras.losses.SparseCategoricalCrossentropy()
      self.loss_3=IoUFocalLoss()

      self.loss_1_tracker = tf.keras.metrics.Mean(name="dice")
      self.loss_2_tracker = tf.keras.metrics.Mean(name="crossentropy")
      self.loss_3_tracker = tf.keras.metrics.Mean(name="iou")

      self.recall_tracker=tf.keras.metrics.Precision(thresholds=0.3,name="Precision")
      self.f1_total_tracker=tf.keras.metrics.BinaryIoU( target_class_ids= [1],threshold=0.3,name="iou_total")
      self.f1_nodamage_tracker     =    tf.keras.metrics.BinaryIoU( target_class_ids= [1],threshold=0.3,name="iou_nodamage")
      self.f1_minordamage_tracker  = tf.keras.metrics.BinaryIoU( target_class_ids= [1],threshold=0.3,name="iou_minordamage")
      self.f1_majordamage_tracker  = tf.keras.metrics.BinaryIoU( target_class_ids= [1],threshold=0.3,name="iou_majordamage")
      self.f1_destroyed_tracker    = tf.keras.metrics.BinaryIoU( target_class_ids= [1],threshold=0.3,name="iou_destroyed")
      self.f1_background_tracker = tf.keras.metrics.BinaryIoU( target_class_ids= [0],threshold=0.3,name="iou_background")

    @property
    def metrics(self):
        return [
            self.loss_1_tracker,
            self.loss_2_tracker,
            self.loss_3_tracker,
            self.f1_total_tracker,
            self.f1_nodamage_tracker,    
            self.f1_minordamage_tracker ,
            self.f1_majordamage_tracker ,
            self.f1_destroyed_tracker   ,
            self.f1_background_tracker,
            self.recall_tracker,
        ]

    def loss3_full_compute(self,y_true, y_pred,weights=None):

        d0=self.loss_3(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d1=self.loss_3(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d2=self.loss_3(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d3=self.loss_3(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d4=self.loss_3(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        if weights:
            d0=d0*0.05
            d1=d1*0.4
            d2=d2*0.6
            d3=d3*0.6
            d4=d4*0.4

        loss=(d1+d2+d3+d4+d0)
        return loss

    def loss1_full_compute(self,y_true, y_pred,weights=None):

        d0=self.loss_1(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d1=self.loss_1(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d2=self.loss_1(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d3=self.loss_1(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d4=self.loss_1(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        if weights:
            d0=d0*0.05
            d1=d1*0.2
            d2=d2*0.6
            d3=d3*0.6
            d4=d4*0.4

        loss=(d1+d2+d3+d4+d0)
        return loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (pre, post),(y_one,y_label) = data

        pre=tf.transpose(pre,(0,3,1,2))
        post=tf.transpose(post,(0,3,1,2))

        with tf.GradientTape() as tape:
            y_pred,y_pred2  = self([pre,post], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss2 = self.loss_2(y_label, y_pred)*4

            loss1 = self.loss_1(y_one, y_pred)
            loss3= self.loss3_full_compute(y_one, y_pred2)*(0.25)
            loss=loss1+(loss2)+loss3

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)



        self.f1_total_tracker.update_state(K.flatten(y_one[:,:,:,1:]),K.flatten(y_pred2[:,:,:,1:]))   
        self.f1_nodamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,1],axis=-1),tf.expand_dims(y_pred2[:,:,:,1],axis=-1))    
        self.f1_minordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,2],axis=-1),tf.expand_dims(y_pred2[:,:,:,2],axis=-1)) 
        self.f1_majordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,3],axis=-1),tf.expand_dims(y_pred2[:,:,:,3],axis=-1)) 
        self.f1_destroyed_tracker.update_state(tf.expand_dims(y_one[:,:,:,4],axis=-1),tf.expand_dims(y_pred2[:,:,:,4],axis=-1))   
        self.f1_background_tracker.update_state(tf.expand_dims(y_one[:,:,:,0],axis=-1),tf.expand_dims(y_pred2[:,:,:,0],axis=-1))
        self.loss_1_tracker.update_state(loss1)
        self.loss_2_tracker.update_state(loss2)
        self.loss_3_tracker.update_state(loss3)

        self.recall_tracker.update_state(K.flatten(tf.expand_dims(y_one[:,:,:,1:],axis=-1)),K.flatten(tf.expand_dims(y_pred2[:,:,:,1:],axis=-1)))
        results = {m.name: m.result() for m in self.metrics}
        return results



    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (pre, post),(y_one,y_label) = data

        pre=tf.transpose(pre,(0,3,1,2))
        post=tf.transpose(post,(0,3,1,2))

        y_pred,y_pred2 = self([pre,post], training=False)  # Forward pass

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss2 = self.loss_2(y_label, y_pred)*4

        loss1 = self.loss_1(y_one, y_pred)
        loss3= self.loss3_full_compute(y_one, y_pred2)*(0.25)
        loss=loss1+(loss2)+loss3

        self.f1_total_tracker.update_state(K.flatten(tf.expand_dims(y_one[:,:,:,1:],axis=-1)),K.flatten(tf.expand_dims(y_pred2[:,:,:,1:],axis=-1)))   
        self.f1_nodamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,1],axis=-1),tf.expand_dims(y_pred2[:,:,:,1],axis=-1))    
        self.f1_minordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,2],axis=-1),tf.expand_dims(y_pred2[:,:,:,2],axis=-1)) 
        self.f1_majordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,3],axis=-1),tf.expand_dims(y_pred2[:,:,:,3],axis=-1)) 
        self.f1_destroyed_tracker.update_state(tf.expand_dims(y_one[:,:,:,4],axis=-1),tf.expand_dims(y_pred2[:,:,:,4],axis=-1))   
        self.f1_background_tracker.update_state(tf.expand_dims(y_one[:,:,:,0],axis=-1),tf.expand_dims(y_pred2[:,:,:,0],axis=-1))
        self.loss_1_tracker.update_state(loss1)
        self.recall_tracker.update_state(K.flatten(tf.expand_dims(y_one[:,:,:,1:],axis=-1)),K.flatten(tf.expand_dims(y_pred2[:,:,:,1:],axis=-1)))
        self.loss_2_tracker.update_state(loss2)
        self.loss_3_tracker.update_state(loss3)

        results = {m.name: m.result() for m in self.metrics}
        return results


def build_model(shape_input,segformer_pretrained,shape=(512,512)):
  input_post= tf.keras.Input(shape=shape_input,name="post_image")
  input_pre = tf.keras.Input(shape=shape_input,name="pre_image")
  output,output2=SegFormerClassifier(segformer_pretrained=segformer_pretrained,shape=shape)(input_pre,input_post)
  model = UsegFormerClass(inputs=[input_pre,input_post], outputs=[output,output2])
  return model



def build_modelv2(shape_input,segformer_pretrained,shape=(512,512)):
  input_post= tf.keras.Input(shape=shape_input,name="post_image")
  input_pre = tf.keras.Input(shape=shape_input,name="pre_image")
  output,output2=SegFormerClassifierV2(segformer_pretrained=segformer_pretrained,shape=shape)(input_pre,input_post)
  model = UsegFormerClass(inputs=[input_pre,input_post], outputs=[output,output2])
  return model
