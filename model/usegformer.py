#ref Huggingg Face Segformer

import tensorflow as tf

import os
from model.loss import *
import os
import datetime
from utils.utils import instantiate_from_config
from model.unet import *
from model.layers import *
import numpy as np
import tensorflow.keras.backend as K

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


import tensorflow as tf

import os
from model.loss import *
import os
import datetime
from utils.utils import instantiate_from_config
from model.unet import *
from model.layers import *
from tensorflow import keras
import tensorflow.keras.backend as K

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

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

    post_input = tf.keras.layers.Input(
        shape=input_shape, name="post_image"
    )
    pre_mask_input = tf.keras.layers.Input(
        shape=context_shape, name="pre_mask"
    )

       

    context_sub1=  tf.keras.layers.Conv2D(16, 3, padding="same", kernel_initializer = 'he_normal')(post_input)
    context_sub2=  tf.keras.layers.Conv2D(16, 3, padding="same", kernel_initializer = 'he_normal')(pre_mask_input)
    context= tf.keras.layers.Concatenate()([context_sub1,context_sub2])

    context = convolution_block(context,num_filters=widths[0])
    context = convolution_block(context,num_filters=widths[0])
    context=  tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(context)
    context = convolution_block(context,num_filters=widths[1])
    context = convolution_block(context,num_filters=widths[1])
    hiddens=[]
    hiddens.append(context)

    shapes=[1,2,3,4]
    for idx,(hidden_pre, hidden_post) in enumerate(zip(hiddenspre, hiddenspost)):

        hidden_pre=DilatedSpatialPyramidPooling(hidden_pre)
        hidden_post=DilatedSpatialPyramidPooling(hidden_post)
        x= tf.keras.layers.Concatenate()([hidden_pre,hidden_post])                   

        for j in range(shapes[idx]):
            for _ in range(num_res_blocks):
              if j==0:
                j=1
              x=convolution_block(x,num_filters=widths[j+2],dilation_rate=j*4)

            x = UpSample(widths[j+2],norm="batchnorm")(x)
        hiddens.append(x)
    x= tf.keras.layers.Concatenate()(hiddens)
    for _ in range(num_res_blocks):
            x=convolution_block(x,num_filters=widths[1])

    x=DilatedSpatialPyramidPooling(x)
    x=convolution_block(x,num_filters=widths[1])
    x=AttentionGate(widths[1],do_upsample=False)(context,x)
    x = UpSample(widths[1],norm="batchnorm")(x)

    for _ in range(num_res_blocks):
            convolution_block(x,num_filters=widths[0])

    context = UpSample(widths[0],norm="batchnorm")(context)

    x=AttentionGate(widths[0],do_upsample=False)(context,x)

    x = tf.keras.layers.Conv2D(widths[0], 3, padding="same", kernel_initializer = 'he_normal')(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(5,1, padding="same", )(x)
    output=tf.keras.layers.Activation("softmax")(x)

    return tf.keras.Model([post_input,pre_mask_input,hiddenspre,hiddenspost], output, name="uspatial_net")



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
        self.loss_2=tf.keras.losses.SparseCategoricalCrossentropy()

        self.loss_2_weighted=WeightedCrossentropy(weights = {0: 1,1: 4})


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

        dices["nodamage"]=d1
        dices["minordamage"]=d2
        dices["majordamage"]=d3
        dices["destroyed"]=d4
        dices["background"]=d0
        dices["total_dice"]= 4/(((d1+1e-6)**-1)+((d2+1e-6)**-1)+((d3+1e-6)**-1)+((d4+1e-6)**-1))
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

    def loss2_full_compute(self,y_true, y_pred):
        d0=self.loss_2(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d1=self.loss_2(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d2=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d3=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d4=self.loss_2_weighted(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))
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
            y_local_pre,hiddens_pre=self.unet_layer(x_pre,training=False)
            y_local_post,hiddens_post=self.unet_layer(x_post,training=False)


            y_multilabel_resized = self.network([x_post,y_local_pre,hiddens_pre,hiddens_post], training=True)
            #upsample_resolution = tf.shape(multilabel_map)
     
            #y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

            loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
            loss_2=self.loss_2(multilabel_map,y_multilabel_resized)*self.loss_weights[1]
            loss=loss_1+loss_2

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        dices=self.dice_classes_score(multilabel_map,y_multilabel_resized)

        self.loss_1_tracker.update_state(loss_1)
        self.loss_2_tracker.update_state(loss_2)
        self.f1_total_tracker.update_state(dices["total_dice"])   
        self.f1_nodamage_tracker.update_state(dices["nodamage"])    
        self.f1_minordamage_tracker.update_state(dices["minordamage"]) 
        self.f1_majordamage_tracker.update_state(dices["majordamage"]) 
        self.f1_destroyed_tracker.update_state(dices["destroyed"])   
        self.f1_background_tracker.update_state(dices["background"])

        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_local_pre,hiddens_pre=self.unet_layer(x_pre,training=False)
        y_local_post,hiddens_post=self.unet_layer(x_post,training=False)
        
        y_multilabel_resized = self.network([x_post,y_local_pre,hiddens_pre,hiddens_post], training=True)
        #upsample_resolution = tf.shape(multilabel_map)
  
        #y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

    

        loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
        loss_2=self.loss_2(multilabel_map,y_multilabel_resized)*self.loss_weights[1]

      
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