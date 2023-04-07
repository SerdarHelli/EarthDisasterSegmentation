from transformers import SegformerConfig,TFSegformerModel,TFSegformerDecodeHead,TFSegformerPreTrainedModel
import math
from typing import Dict, Tuple
from typing import Optional
import tensorflow as tf

from model.layers import *


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

        self.diff0 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_diff_0")
        self.pred0 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_0")
        self.diff1 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_diff_1")
        self.pred1 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_1")
        self.diff2 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_diff_2")
        self.pred2 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_2")
        self.diff3 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_diff_3")
        self.pred3 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_3")
        self.res0 =ResBlock(256//2)
        self.upsample0 =UpSample(256//2)
        self.res1 =ResBlock(128//2)
        self.upsample1 =UpSample(128//2)
        self.res2 =ResBlock(64//2)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse",activity_regularizer=tf.keras.regularizers.L2(1e-6)
        )

        self.linear_fuse_post=[]
        self.linear_fuse_post .append( tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size//4, kernel_size=1, use_bias=False, name="linear_fuse_post",activity_regularizer=tf.keras.regularizers.L2(1e-6)
        ))

        self.linear_fuse_post.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
        self.linear_fuse_post.append(tf.keras.layers.Activation("relu"))
        self.linear_fuse_post.append( tf.keras.layers.Conv2D(
              filters=config.num_labels, kernel_size=7,  padding="same" , kernel_initializer = 'he_normal'
          ))

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        self.activation = tf.keras.layers.Activation("relu")

        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)

        self.configuration=[]
        self.configuration_first=tf.keras.layers.AveragePooling2D(pool_size=(16, 16))

        self.configuration.append( tf.keras.layers.Conv2D(128,(7,7),padding="same",  kernel_initializer = 'he_normal'))
        self.configuration.append(tf.keras.layers.BatchNormalization())
        self.configuration.append(tf.keras.layers.Activation("relu"))
        self.configuration.append(tf.keras.layers.Dropout(0.5))

        self.configuration.append(tf.keras.layers.Conv2D(128,(7,7), padding="same" , kernel_initializer = 'he_normal'))
        self.configuration.append(tf.keras.layers.BatchNormalization())
        self.configuration.append(tf.keras.layers.Activation("relu"))
        self.configuration.append(tf.keras.layers.Dropout(0.5))

        self.configuration.append(tf.keras.layers.Conv2D(128,(7,7),padding="same", kernel_initializer = 'he_normal'))
        self.configuration.append(tf.keras.layers.BatchNormalization())
        self.configuration.append(tf.keras.layers.Activation("relu"))
        self.configuration.append(tf.keras.layers.Dropout(0.5))

        self.configuration.append(tf.keras.layers.Conv2D(config.num_labels,(7,7), activation = "sigmoid", padding="same", kernel_initializer =tf.keras.initializers.RandomNormal(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
        self.configuration.append(tf.keras.layers.UpSampling2D(size=(16, 16),interpolation='bilinear'))

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

        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
    
        # logits of shape (batch_size, height/4, width/4, num_labels)
        logits = self.classifier(hidden_states)

        c=[tf.image.resize(a, size=(tf.shape(logits)[1]//16,tf.shape(logits)[1]//16), method="bilinear")
           for a in all_hidden_states_post
           ]

        post_hiddens=tf.concat(c, axis=-1)
        for block in self.linear_fuse_post:
            post_hiddens=block(post_hiddens)


        x_sub=self.configuration_first(logits)
        x=tf.concat([post_hiddens,x_sub], axis=-1)

        for block in self.configuration:
            x=block(x)
        
        logits2=logits*x

        return logits,logits2

class SegFormerClassifier(tf.keras.Model):
    def __init__(self, segformer: Optional[TFSegformerModel] = None):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
  
        self.segformer: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        #self.segformer_post: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        self.config.num_labels = 5
        #self.config.hidden_sizes = [hidden_size * 2 for hidden_size in self.config.hidden_sizes]
        self.decode_head: SegformerDecodeHead = SegformerDecodeHead(self.config)
        self.segformer_input_size = (512, 512)



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

        logits,logits2 = self.decode_head(pre_outputs.hidden_states, post_outputs.hidden_states)

        return tf.nn.softmax(logits),tf.nn.softmax(logits2)
    

class SegFormerClassifier(tf.keras.Model):
    def __init__(self, segformer_pretrained,shape=(512,512)):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained(segformer_pretrained)
  
        self.segformer: TFSegformerModel = TFSegformerModel.from_pretrained(segformer_pretrained)
        #self.segformer_post: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        self.config.num_labels = 5
        #self.config.hidden_sizes = [hidden_size * 2 for hidden_size in self.config.hidden_sizes]
        self.decode_head: SegformerDecodeHead = SegformerDecodeHead(self.config)
        self.segformer_input_size = shape



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

        logits,logits2 = self.decode_head(pre_outputs.hidden_states, post_outputs.hidden_states)

        return tf.nn.softmax(logits),tf.nn.softmax(logits2)
    




class SegformerDecodeHeadV2(TFSegformerPreTrainedModel):
    def __init__(self,config, **kwargs):
        super().__init__( config,**kwargs)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        self.config=config
        mlps = []
 
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config, name=f"linear_c.{i}")
            mlps.append(mlp)

        self.mlps = mlps

        self.diff0 = Difference(256,drop_path_rate=0.1,name="conv_decoder_diff_0")
        self.pred0 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_0")
        self.diff1 = Difference(256,drop_path_rate=0.1,name="conv_decoder_diff_1")
        self.pred1 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_1")
        self.diff2 = Difference(256,drop_path_rate=0.1,name="conv_decoder_diff_2")
        self.pred2 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_2")
        self.diff3 = Difference(256,drop_path_rate=0.1,name="conv_decoder_diff_3")
        self.pred3 = ConvBlock(256,drop_path_rate=0.1,name="conv_decoder_pred_3")
        self.res0 =ResBlock(256//2)
        self.upsample0 =UpSample(256//2)
        self.res1 =ResBlock(128//2)
        self.upsample1 =UpSample(128//2)
        self.res2 =ResBlock(64//2)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse",activity_regularizer=tf.keras.regularizers.L2(1e-6)
        )

        self.linear_fuse_post=[]
        self.linear_fuse_post .append( tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size//4, kernel_size=1, use_bias=False, name="linear_fuse_post",activity_regularizer=tf.keras.regularizers.L2(1e-6)
        ))

        self.linear_fuse_post.append(tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9))
        self.linear_fuse_post.append(tf.keras.layers.Activation("relu"))
        self.linear_fuse_post.append( tf.keras.layers.Conv2D(
              filters=config.num_labels, kernel_size=7,  padding="same" , kernel_initializer = 'he_normal'
          ))

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        self.activation = tf.keras.layers.Activation("relu")

        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)

        self.configuration=[]
        self.configuration_first=tf.keras.layers.AveragePooling2D(pool_size=(16, 16))

        self.configuration.append( tf.keras.layers.Conv2D(128,(7,7),padding="same",  kernel_initializer = 'he_normal'))
        self.configuration.append(tf.keras.layers.BatchNormalization())
        self.configuration.append(tf.keras.layers.Activation("relu"))
        self.configuration.append(tf.keras.layers.Dropout(0.5))

        self.configuration.append(tf.keras.layers.Conv2D(128,(7,7), padding="same" , kernel_initializer = 'he_normal'))
        self.configuration.append(tf.keras.layers.BatchNormalization())
        self.configuration.append(tf.keras.layers.Activation("relu"))
        self.configuration.append(tf.keras.layers.Dropout(0.5))

        self.configuration.append(tf.keras.layers.Conv2D(128,(7,7),padding="same", kernel_initializer = 'he_normal'))
        self.configuration.append(tf.keras.layers.BatchNormalization())
        self.configuration.append(tf.keras.layers.Activation("relu"))
        self.configuration.append(tf.keras.layers.Dropout(0.5))

        self.configuration.append(tf.keras.layers.Conv2D(config.num_labels,(7,7), activation = "sigmoid", padding="same", kernel_initializer =tf.keras.initializers.RandomNormal(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005)))
        self.configuration.append(tf.keras.layers.UpSampling2D(size=(16, 16),interpolation='bilinear'))

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
        hidden_states_0x=self.diff0(hidden_states_0)
        hidden_states_0=self.pred0(hidden_states_0x)
        hidden_states_0 = tf.image.resize(hidden_states_0, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_0)


        hidden_states_1=tf.concat([all_hidden_states_pre[1],all_hidden_states_post[1]],axis=-1)
        hid_sub1= tf.image.resize(hidden_states_0, size=(tf.shape(hidden_states_1)[1],tf.shape(hidden_states_1)[2]), method="bilinear")
        hidden_states_1x=self.diff1(hidden_states_1)+hid_sub1
        hidden_states_1=self.pred1(hidden_states_1x)
        hidden_states_1 = tf.image.resize(hidden_states_1, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_1)

        hidden_states_2=tf.concat([all_hidden_states_pre[2],all_hidden_states_post[2]],axis=-1)
        hid_sub2= tf.image.resize(hidden_states_1, size=(tf.shape(hidden_states_2)[1],tf.shape(hidden_states_2)[2]), method="bilinear")
        hidden_states_2x=self.diff2(hidden_states_2)+hid_sub2
        hidden_states_2=self.pred2(hidden_states_2x)
        hidden_states_2 = tf.image.resize(hidden_states_2, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_2)

        hidden_states_3=tf.concat([all_hidden_states_pre[3],all_hidden_states_post[3]],axis=-1)
        hid_sub3= tf.image.resize(hidden_states_2, size=(tf.shape(hidden_states_3)[1],tf.shape(hidden_states_3)[2]), method="bilinear")
        hidden_states_3x=self.diff3(hidden_states_3)+hid_sub3
        hidden_states_3=self.pred3(hidden_states_3x)
        hidden_states_3 = tf.image.resize(hidden_states_3, size=upsample_resolution, method="bilinear")
        all_hidden_states.append(hidden_states_3)



        hidden_states = self.linear_fuse(tf.concat(all_hidden_states, axis=-1))
        hidden_states=self.res0(hidden_states)
        hidden_states=self.upsample0(hidden_states)
        hidden_states=self.res1(hidden_states)
        hidden_states=self.upsample1(hidden_states)
        hidden_states=self.res2(hidden_states)

        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
    
        # logits of shape (batch_size, height/4, width/4, num_labels)
        logits = self.classifier(hidden_states)


        all_difference_hidden_state=[hidden_states_0x,hidden_states_1x,hidden_states_2x,hidden_states_3x]
        c=[tf.image.resize(a, size=(tf.shape(logits)[1]//16,tf.shape(logits)[1]//16), method="bilinear")
           for a in all_difference_hidden_state
           ]

        post_hiddens=tf.concat(c, axis=-1)
        for block in self.linear_fuse_post:
            post_hiddens=block(post_hiddens)


        x_sub=self.configuration_first(logits)
        x=tf.concat([post_hiddens,x_sub], axis=-1)

        for block in self.configuration:
            x=block(x)
        
        logits2=logits*x

        return logits,logits2

class SegFormerClassifierV2(tf.keras.Model):
    def __init__(self, segformer: Optional[TFSegformerModel] = None):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
  
        self.segformer: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        #self.segformer_post: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        self.config.num_labels = 5
        #self.config.hidden_sizes = [hidden_size * 2 for hidden_size in self.config.hidden_sizes]
        self.decode_head: SegformerDecodeHead = SegformerDecodeHeadV2(self.config)
        self.segformer_input_size = (512, 512)



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

        logits,logits2 = self.decode_head(pre_outputs.hidden_states, post_outputs.hidden_states)

        return tf.nn.softmax(logits),tf.nn.softmax(logits2)
    

class SegFormerClassifier(tf.keras.Model):
    def __init__(self, segformer_pretrained,shape=(512,512)):
        super().__init__()
        self.config: SegformerConfig = SegformerConfig.from_pretrained(segformer_pretrained)
  
        self.segformer: TFSegformerModel = TFSegformerModel.from_pretrained(segformer_pretrained)
        #self.segformer_post: TFSegformerModel = TFSegformerModel.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        self.config.num_labels = 5
        #self.config.hidden_sizes = [hidden_size * 2 for hidden_size in self.config.hidden_sizes]
        self.decode_head: SegformerDecodeHead = SegformerDecodeHead(self.config)
        self.segformer_input_size = shape



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

        logits,logits2 = self.decode_head(pre_outputs.hidden_states, post_outputs.hidden_states)

        return tf.nn.softmax(logits),tf.nn.softmax(logits2)
    
