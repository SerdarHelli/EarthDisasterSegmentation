from transformers import SegformerConfig,TFSegformerModel,TFSegformerPreTrainedModel

from typing import Optional
from model.layers import *
import tensorflow as tf
import tensorflow.keras.backend as K
from model.loss import *

class GetDifference(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.filters=input_shape[1]
        self.proj =tf.keras.layers.Conv2D(self.filters,kernel_size=7,padding="same",use_bias=False, kernel_initializer='he_normal')
        self.proj1 =tf.keras.layers.GlobalAveragePooling2D()
        self.proj2 =tf.keras.layers.Dense(self.filters)

    def call(self, input_post_tensor: tf.Tensor,input_pre_tensor:tf.Tensor):
        diff=tf.math.abs(input_post_tensor-input_pre_tensor)

        att=tf.concat([input_post_tensor,input_pre_tensor,diff],axis=1)
        att=tf.transpose(att,(0, 2, 3, 1))
        att=self.proj(att)
        att=self.proj1(att)
        att=tf.nn.sigmoid(self.proj2(att))

        att=tf.reshape(att,(tf.shape(input_post_tensor)[0],self.filters,1,1))
        return diff*att


class TFSegformerMLP(tf.keras.layers.Layer):
    """
    Linear Embedding.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        decoder_hidden_size=config.decoder_hidden_size
        self.proj = tf.keras.layers.Dense(decoder_hidden_size)

    def call(self, hidden_states: tf.Tensor):
        height = tf.shape(hidden_states)[1]
        width = tf.shape(hidden_states)[2]
        hidden_dim = tf.shape(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (-1, height * width, hidden_dim))
        hidden_states = self.proj(hidden_states)

        return hidden_states



class SegformerDecodeHead(TFSegformerPreTrainedModel):
    def __init__(self,config, **kwargs):
        super().__init__( config,**kwargs)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        self.config=config
        mlps = []
 
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config, name=f"linear_c.{i}")
            mlps.append(mlp)

        self.mlps = mlps

        self.diff0 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_diff_0")
        self.pred0 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_pred_0")
        self.diff1 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_diff_1")
        self.pred1 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_pred_1")
        self.diff2 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_diff_2")
        self.pred2 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_pred_2")
        self.diff3 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_diff_3")
        self.pred3 = ConvBlock(config.decoder_hidden_size,drop_path_rate=0.1,name="conv_decoder_pred_3")
        self.res0 =ResBlock(256)
        self.upsample0 =UpSample(256)
        self.res1 =ResBlock(128)
        self.upsample1 =UpSample(128)
        self.res2 =ResBlock(64)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse",activity_regularizer=tf.keras.regularizers.L2(1e-6)
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        self.activation = tf.keras.layers.Activation("relu")

        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)



        self.classifier = tf.keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name="classifier")


    def call(self, encoder_hidden_states_pre,encoder_hidden_states_post, training: bool = True):
        batch_size = tf.shape(encoder_hidden_states_pre[-1])[0]

        all_hidden_states_pre = []
        all_hidden_states_post= []
        all_hidden_states=[]
        upsample_resolution = tf.shape( tf.transpose(encoder_hidden_states_pre[0], perm=[0, 2, 3, 1]))[1:-1]

        for idx,(encoder_hidden_state_pre, encoder_hidden_state_post,mlp) in enumerate(zip(encoder_hidden_states_pre, encoder_hidden_states_post,self.mlps)):
            if self.config.reshape_last_stage is False and len(tf.shape(encoder_hidden_state_pre)) == 3:
                height = tf.math.sqrt(tf.cast(tf.shape(encoder_hidden_state_pre)[1], tf.float32))
                height = width = tf.cast(height, tf.int32)
                encoder_hidden_state_pre = tf.reshape(encoder_hidden_state_pre, (batch_size, height, width, -1))
                encoder_hidden_state_post = tf.reshape(encoder_hidden_state_post, (batch_size, height, width, -1))

            # unify channel dimension
            

            encoder_hidden_state_pre = tf.transpose(encoder_hidden_state_pre, perm=[0, 2, 3, 1])
            encoder_hidden_state_post = tf.transpose(encoder_hidden_state_post, perm=[0, 2, 3, 1])

            height = tf.shape(encoder_hidden_state_post)[1]
            width = tf.shape(encoder_hidden_state_post)[2]
            encoder_hidden_state_post = mlp(encoder_hidden_state_post)
            encoder_hidden_state_pre = mlp(encoder_hidden_state_pre)
            encoder_hidden_state_pre = tf.reshape(encoder_hidden_state_pre, (batch_size, height, width, tf.shape(encoder_hidden_state_pre)[-1]))
            encoder_hidden_state_post = tf.reshape(encoder_hidden_state_post, (batch_size, height, width, tf.shape(encoder_hidden_state_post)[-1]))
            all_hidden_states_pre.append(encoder_hidden_state_pre)
            all_hidden_states_post.append(encoder_hidden_state_post)


        hidden_states_0=tf.concat([all_hidden_states_pre[0],all_hidden_states_post[0]],axis=-1)
        hidden_states_0=self.diff0(hidden_states_0)
        hidden_states_0=self.pred0(hidden_states_0)
        hidden_states_0 = tf.image.resize(hidden_states_0, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_0)


        hidden_states_1=tf.concat([all_hidden_states_pre[1],all_hidden_states_post[1]],axis=-1)
        hid_sub1= tf.image.resize(hidden_states_0, size=(tf.shape(hidden_states_1)[1],tf.shape(hidden_states_1)[2]), method="bilinear")
        hidden_states_1=self.diff1(hidden_states_1)+hid_sub1
        hidden_states_1=self.pred1(hidden_states_1)
        hidden_states_1 = tf.image.resize(hidden_states_1, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_1)

        hidden_states_2=tf.concat([all_hidden_states_pre[2],all_hidden_states_post[2]],axis=-1)
        hid_sub2= tf.image.resize(hidden_states_1, size=(tf.shape(hidden_states_2)[1],tf.shape(hidden_states_2)[2]), method="bilinear")
        hidden_states_2=self.diff2(hidden_states_2)+hid_sub2
        hidden_states_2=self.pred2(hidden_states_2)
        hidden_states_2 = tf.image.resize(hidden_states_2, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_2)

        hidden_states_3=tf.concat([all_hidden_states_pre[3],all_hidden_states_post[3]],axis=-1)
        hid_sub3= tf.image.resize(hidden_states_2, size=(tf.shape(hidden_states_3)[1],tf.shape(hidden_states_3)[2]), method="bilinear")
        hidden_states_3=self.diff3(hidden_states_3)+hid_sub3
        hidden_states_3=self.pred3(hidden_states_3)
        hidden_states_3 = tf.image.resize(hidden_states_3, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_3)



        hidden_states = self.linear_fuse(tf.concat(all_hidden_states, axis=-1))
        hidden_states=self.res0(hidden_states)
        hidden_states=self.upsample0(hidden_states)
        hidden_states=self.res1(hidden_states)
        hidden_states=self.upsample1(hidden_states)
        hidden_states=self.res2(hidden_states)

        hidden_states = self.batch_norm(hidden_states, training=training)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
    
        # logits of shape (batch_size, height/4, width/4, num_labels)
        logits = self.classifier(hidden_states)

        return logits
    
    
class SegFormerClassifier(tf.keras.Model):
    def __init__(self, segformer: Optional[TFSegformerModel] = None):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
  
        self.segformer: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        #self.segformer_post: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        self.config.num_labels = 5
        self.config.hidden_sizes = [hidden_size * 2 for hidden_size in self.config.hidden_sizes]
        self.decode_head: SegformerDecodeHead = SegformerDecodeHead(self.config)
        self.segformer_input_size = (512, 512)
        diffs = []
        for i in range(self.config.num_encoder_blocks):
            diff = GetDifference( name=f"difference_c.{i}")
            diffs.append(diff)
        self.diffs = diffs


    def call(self, pre,post) :
        return_dict: bool = True
        output_hidden_states: bool = False


        pre_outputs = self.segformer(
            pre,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        post_outputs = self.segformer(
            post,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        logits = self.decode_head(pre_outputs.hidden_states, post_outputs.hidden_states)

        return tf.nn.sigmoid(logits)
    
class CustomModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      
      self.loss_1= DiceFocalLoss()

      self.loss_1_tracker = tf.keras.metrics.Mean(name="loss")
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
            self.f1_total_tracker,
            self.f1_nodamage_tracker,    
            self.f1_minordamage_tracker ,
            self.f1_majordamage_tracker ,
            self.f1_destroyed_tracker   ,
            self.f1_background_tracker,
            self.recall_tracker,
        ]

    def loss1_full_compute(self,y_true, y_pred,weights=None):

        d0=self.loss_1(tf.expand_dims(y_true[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        d1=self.loss_1(tf.expand_dims(y_true[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))
        d2=self.loss_1(tf.expand_dims(y_true[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1))
        d3=self.loss_1(tf.expand_dims(y_true[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1))
        d4=self.loss_1(tf.expand_dims(y_true[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))

        if weights:
            d0=d0*0.05
            d1=d1*0.2
            d2=d2*0.4
            d3=d3*0.6
            d4=d4*0.4

        loss=(d1+d2+d3+d4+d0)
        return loss


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (pre, post),(y_one,_) = data

        pre=tf.transpose(pre,(0,3,1,2))
        post=tf.transpose(post,(0,3,1,2))

        with tf.GradientTape() as tape:
            y_pred = self([pre,post], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            loss = self.loss_1(y_one, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)



        self.f1_total_tracker.update_state(K.flatten(y_one[:,:,:,1:]),K.flatten(y_pred[:,:,:,1:]))   
        self.f1_nodamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))    
        self.f1_minordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1)) 
        self.f1_majordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1)) 
        self.f1_destroyed_tracker.update_state(tf.expand_dims(y_one[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))   
        self.f1_background_tracker.update_state(tf.expand_dims(y_one[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        self.loss_1_tracker.update_state(loss)
        self.recall_tracker.update_state(K.flatten(tf.expand_dims(y_one[:,:,:,1:],axis=-1)),K.flatten(tf.expand_dims(y_pred[:,:,:,1:],axis=-1)))
        results = {m.name: m.result() for m in self.metrics}
        return results



    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        (pre, post),(y_one,_) = data

        pre=tf.transpose(pre,(0,3,1,2))
        post=tf.transpose(post,(0,3,1,2))

        y_pred = self([pre,post], training=False)  # Forward pass

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        loss = self.loss_1(y_one, y_pred)


        self.f1_total_tracker.update_state(K.flatten(tf.expand_dims(y_one[:,:,:,1:],axis=-1)),K.flatten(tf.expand_dims(y_pred[:,:,:,1:],axis=-1)))   
        self.f1_nodamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,1],axis=-1),tf.expand_dims(y_pred[:,:,:,1],axis=-1))    
        self.f1_minordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,2],axis=-1),tf.expand_dims(y_pred[:,:,:,2],axis=-1)) 
        self.f1_majordamage_tracker.update_state(tf.expand_dims(y_one[:,:,:,3],axis=-1),tf.expand_dims(y_pred[:,:,:,3],axis=-1)) 
        self.f1_destroyed_tracker.update_state(tf.expand_dims(y_one[:,:,:,4],axis=-1),tf.expand_dims(y_pred[:,:,:,4],axis=-1))   
        self.f1_background_tracker.update_state(tf.expand_dims(y_one[:,:,:,0],axis=-1),tf.expand_dims(y_pred[:,:,:,0],axis=-1))
        self.loss_1_tracker.update_state(loss)
        self.recall_tracker.update_state(K.flatten(tf.expand_dims(y_one[:,:,:,1:],axis=-1)),K.flatten(tf.expand_dims(y_pred[:,:,:,1:],axis=-1)))

        results = {m.name: m.result() for m in self.metrics}
        return results

def build_usegformer(shape_input):
  input_post= tf.keras.Input(shape=shape_input,name="post_image")
  input_pre = tf.keras.Input(shape=shape_input,name="pre_image")
  output=SegFormerClassifier()(input_pre,input_post)
  model = CustomModel(inputs=[input_pre,input_post], outputs=[output])
  return model