#ref Huggingg Face Segformer

import tensorflow as tf
import math
from typing import Optional
from tensorflow import keras
from tensorflow.keras import layers
import os
from model.loss import *
import os
import datetime
from utils.utils import instantiate_from_config
from model.unet import *
import tensorflow.keras.backend as K

class TFSegformerDropPath(tf.keras.layers.Layer):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super().__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

class TFSegformerOverlapPatchEmbeddings(tf.keras.layers.Layer):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Conv2D(
            filters=hidden_size, kernel_size=patch_size, strides=stride, padding="same"
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05,)

    def call(self, inputs):
        embeddings = self.proj(inputs)
        shape=tf.shape(embeddings)

        height = shape[1]
        width = shape[2]
        hidden_dim = shape[3]
        # (batch_size, height, width, num_channels) -> (batch_size, height*width, num_channels)
        # this can be fed to a Transformer layer
        embeddings = tf.reshape(embeddings, (shape[0], height * width, hidden_dim))
        embeddings = self.layer_norm(embeddings)
        return embeddings,height, width

class TFSegformerEfficientSelfAttention(tf.keras.layers.Layer):
    """SegFormer's efficient self-attention mechanism. Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        attention_probs_dropout_prob:int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(self.all_head_size, name="query")
        self.key = tf.keras.layers.Dense(self.all_head_size, name="key")
        self.value = tf.keras.layers.Dense(self.all_head_size, name="value")

        self.dropout = tf.keras.layers.Dropout(attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(
                filters=hidden_size, kernel_size=sequence_reduction_ratio, strides=sequence_reduction_ratio, name="sr"
            )
            self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")


    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size]
        # to [batch_size, seq_length, num_attention_heads, attention_head_size]
        batch_size = tf.shape(tensor)[0]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size]
        # to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        hidden_states: tf.Tensor,
        mask: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = True,
    ) :
        batch_size = tf.shape(hidden_states)[0]
        num_channels = tf.shape(hidden_states)[2]

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if tf.shape(mask)[1]!=tf.shape(hidden_states)[1]:
            mask=tf.image.resize(mask,(height * width, num_channels))

        if self.sr_ratio > 1:
            mask = tf.reshape(mask, (batch_size, height, width, num_channels))
            mask = self.sr(mask)      
            mask = tf.reshape(mask, (batch_size, -1, num_channels))
            mask = self.layer_norm(mask)

        key_layer = self.transpose_for_scores(self.key(mask))
        value_layer = self.transpose_for_scores(self.value(mask))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)

        scale = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, scale)

        # Normalize the attention scores to probabilities.
        attention_probs = tf.nn.softmax(logits=attention_scores + 1e-9, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)

        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, all_head_size)
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class TFSegformerSelfOutput(tf.keras.layers.Layer):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(hidden_size, name="dense")

    def call(self, hidden_states: tf.Tensor,  training: bool = True) -> tf.Tensor:

        hidden_states = self.dense(hidden_states)
        return hidden_states


class TFSegformerAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        sequence_reduction_ratio: int,
        attention_probs_dropout_prob:int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.self = TFSegformerEfficientSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            name="self",
        )
        self.dense_output = TFSegformerSelfOutput( hidden_size=hidden_size,name="output")

    def call(
        self, hidden_states: tf.Tensor,  mask: tf.Tensor, height: int, width: int, output_attentions: bool = False
    ) :
        self_outputs = self.self(hidden_states,mask, height, width, output_attentions)

        attention_output = self.dense_output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TFSegformerDWConv(tf.keras.layers.Layer):
    def __init__(self, dim: int = 768, **kwargs):
        super().__init__(**kwargs)
        self.depthwise_convolution = tf.keras.layers.Conv2D(
            filters=dim, kernel_size=3, strides=1, padding="same", groups=dim, name="dwconv"
        )

    def call(self, hidden_states: tf.Tensor, height: int, width: int):
        batch_size = tf.shape(hidden_states)[0]
        num_channels = tf.shape(hidden_states)[-1]
        hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
        hidden_states = self.depthwise_convolution(hidden_states)

        new_height = tf.shape(hidden_states)[1]
        new_width = tf.shape(hidden_states)[2]
        num_channels = tf.shape(hidden_states)[3]
        hidden_states = tf.reshape(hidden_states, (batch_size, new_height * new_width, num_channels))
        return hidden_states
    



class Gelu(tf.keras.layers.Layer):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
    def call(self,inputs):
        cdf = 0.5 * (1.0 + tf.math.erf(inputs / tf.cast(tf.sqrt(2.0), inputs.dtype)))
        return inputs * cdf



class TFSegformerMixFFN(tf.keras.layers.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        out_features = out_features or in_features
        self.dense1 = tf.keras.layers.Dense(hidden_features, name="dense1")
        self.depthwise_convolution = TFSegformerDWConv(hidden_features, name="dwconv")

        self.intermediate_act_fn = Gelu()

        self.dense2 = tf.keras.layers.Dense(out_features, name="dense2")

    def call(self, hidden_states: tf.Tensor, height: int, width: int, training: bool = True) :
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.depthwise_convolution(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states

class TFSegformerMLP(tf.keras.layers.Layer):
    """
    Linear Embedding.
    """

    def __init__(self, config,dropout=0, **kwargs):
        super().__init__(**kwargs)
        decoder_hidden_size=config.decoder_hidden_size
        self.norm=tf.keras.layers.LayerNormalization()
        self.proj = tf.keras.layers.Dense(decoder_hidden_size, name="proj")
        self.dropout_layer=None
        if dropout>0:
            self.dropout_layer=tf.keras.layers.Dropout(dropout)

    def call(self, hidden_states: tf.Tensor):
        hidden_states=self.norm(hidden_states)
        hidden_states = self.proj(hidden_states)
        if self.dropout_layer:
            hidden_states=self.dropout_layer(hidden_states)
        return hidden_states


class TFSegformerLayer(tf.keras.layers.Layer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequence_reduction_ratio: int,
        mlp_ratio: int,
        attention_probs_dropout_prob:int,
        patch_size:int,
        stride:int,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_1")
        self.attention = TFSegformerAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            name="attention",
        )
        self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else tf.keras.layers.Activation("linear")
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm_2")
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TFSegformerMixFFN( in_features=hidden_size, hidden_features=mlp_hidden_size, name="mlp")
        self.patch_embedd=TFSegformerOverlapPatchEmbeddings(
                            patch_size=patch_size,
                            stride=stride,
                            hidden_size=hidden_size,
                        )
        
    def call(
        self,
        hidden_states: tf.Tensor,
        mask: tf.Tensor,
        output_attentions: bool = False,
        training: bool = True,
    ) :
        mask,height, width=self.patch_embedd(mask)

        hidden_states=tf.reshape(hidden_states,(tf.shape(hidden_states)[0],height*width,tf.shape(hidden_states)[-1]))
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states), # in Segformer, layernorm is applied before self-attention
            mask, 
            height,
            width,
            output_attentions=output_attentions,
            training=training,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output, training=training)
        hidden_states = attention_output + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output, training=training)
        layer_output = mlp_output + hidden_states


        layer_output=tf.reshape(layer_output,(tf.shape(hidden_states)[0],height,width,tf.shape(hidden_states)[-1]))

        return layer_output
    
    
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters,drop_path_rate=0,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
        self.drop_path_rate=drop_path_rate

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
        self.norm2 = getNorm(self.norm_str)
        self.droput=tf.keras.layers.Dropout(self.drop_path_rate)


    def call(self, input_tensor: tf.Tensor):
        x = self.norm1(input_tensor)
        x = self.conv_1(tf.nn.relu(x))
        x = self.norm2(x)
        x = self.conv_2(tf.nn.relu(x))
        x=self.droput(x)
        return x

def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
        )(x)
        return x

    return apply

def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same"
        )(x)
        return x

    return apply
def getNorm(norm_str,eps=1e-6):
    x=None
    if norm_str=="batchnorm":
        x= tf.keras.layers.BatchNormalization(epsilon=eps)
    if norm_str=="layernorm":
        x= tf.keras.layers.LayerNormalization(epsilon=eps)
    if not x:
        raise("Invalid Normalization ")
    return x


def build_usegformernet_model(
    input_shape,
    context_shape,
    widths,
    num_attention_heads,
    sr_ratios,
    has_attention,
    strides,
    num_res_blocks=2,
    activation_fn=keras.activations.swish,
    first_conv_channels=16,
):


    image_input = layers.Input(
        shape=input_shape, name="image_post"
    )
    context= layers.Input(
        shape=context_shape, name="image_pre"
    )

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=3,
        padding="same",
    )(image_input)



    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ConvBlock(
                widths[i]
            )(x)
            if has_attention[i]:

                  x=TFSegformerLayer(
                        hidden_size=widths[i],
                        num_attention_heads=num_attention_heads[i],
                        drop_path=0.1,
                        patch_size=(len(widths)-i)*4,
                        stride=strides[i],
                        sequence_reduction_ratio=sr_ratios[i],
                        mlp_ratio=1,
                        attention_probs_dropout_prob=0.1,
                    )(x,context)


            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ConvBlock(widths[-1] )(
        x, 
    )

    x =   TFSegformerLayer(
                        hidden_size=widths[-1],
                        num_attention_heads=num_attention_heads[i],
                        drop_path=0.1,
                        patch_size=4,
                        stride=strides[-1],
                        sequence_reduction_ratio=sr_ratios[i],
                        mlp_ratio=1,
                        attention_probs_dropout_prob=0.1,
                    )(x,context)


    x = ConvBlock(widths[-1])(
        x,
    )

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ConvBlock(
                widths[i]
            )(x)
            if has_attention[i]:
                  x=TFSegformerLayer(
                        hidden_size=widths[i],
                        num_attention_heads=num_attention_heads[i],
                        drop_path=0.1,
                        patch_size=(len(widths)-i)*4,
                        stride=strides[i],
                        sequence_reduction_ratio=sr_ratios[i],
                        mlp_ratio=1,
                        attention_probs_dropout_prob=0.1,
                    )(x,context)


        if i != 0:
            x = UpSample(widths[i])(x)

    # End block
    x=getNorm("batchnorm")(x)
    x = activation_fn(x)
    x = layers.Conv2D(5, (3, 3), padding="same")(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    return keras.Model([image_input, context], x, name="usegformer_net")

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
        #self.unet_layer=None
        #self.load_unetmodel(unet_config,unet_checkpoint_path)
        #self.segformer_layer = TFSegformerForSemanticSegmentation(config)
        self.network=instantiate_from_config(config.usegformer)
        self.threshold_metric=config.threshold_metric
        self.loss_weights=config.loss_weights
        self.loss_1_tracker = tf.keras.metrics.Mean(name="Dice_loss")
        self.loss_2_tracker = tf.keras.metrics.Mean(name="Crossentropy_Loss")
        self.iou_score_tracker= tf.keras.metrics.Mean(name="Meaniou")
        self.f1_total_tracker=tf.keras.metrics.Mean(name="f1_total")
        self.f1_nodamage_tracker     =    tf.keras.metrics.Mean(name="f1_nodamage")
        self.f1_minordamage_tracker  = tf.keras.metrics.Mean(name="f1_minordamage")
        self.f1_majordamage_tracker  = tf.keras.metrics.Mean(name="f1_majordamage")
        self.f1_destroyed_tracker    = tf.keras.metrics.Mean(name="f1_destroyed")
        self.f1_unclassified_tracker = tf.keras.metrics.Mean(name="f1_unclassified")
        self.iou_score2_tracker=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1],name="iou")
        self.class_weights=config.weights


        self.checkpoint_dir = os.path.join(checkpoint_path,"checkpoint")
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("New Checkpoint Folder Initialized...")
            
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.special_checkpoint=special_checkpoint
    
    # def load_unetmodel(self,unet_config,unet_checkpoint_path):
    #     unet_model=UNetModel(unet_config,checkpoint_path=unet_checkpoint_path)
    #     unet_model.compile()
    #     print("Loading Unet Model")
    #     unet_model.load(usage="eval")
    #     layer_names=[layer.name for layer in unet_model.network.layers]

    #     self.unet_layer=unet_model.network.get_layer(layer_names[-1])
    #     del unet_model

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
        self.loss_2=tf.keras.losses.BinaryCrossentropy()
        self.iou_score=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score1=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score2=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score3=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score4=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])
        self.iou_score5=tf.keras.metrics.BinaryIoU(threshold=self.threshold_metric,target_class_ids=[1])

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
            self.iou_score2_tracker,
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
        dices["unclassified"]=(2*d5)/(1+d5)
        dices["total_dice"]= (dices["nodamage"]+dices["minordamage"]+dices["majordamage"]+dices["unclassified"])/5
        return dices
    
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
        loss=(d1+d2+d3+d4+d5)/5
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

            y_multilabel_resized = self.network([x_pre,x_post], training=True)
            #upsample_resolution = tf.shape(multilabel_map)
     
            #y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

            loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
            loss_2=self.loss2_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[1]
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
        self.iou_score2_tracker.update_state(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))

        results = {m.name: m.result() for m in self.metrics}
        return results


    def test_step(self, inputs):
        # 1. Get the batch size
        (x_pre,x_post),(local_map,multilabel_map)=inputs

        y_multilabel_resized = self.network([x_pre,x_post], training=False)
        #upsample_resolution = tf.shape(multilabel_map)
  
        #y_multilabel_resized = tf.image.resize(y_multilabel, size=(upsample_resolution[1],upsample_resolution[2]), method="bilinear")

    

        loss_1=self.loss1_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[0]
        loss_2=self.loss2_full_compute(multilabel_map,y_multilabel_resized,weights=self.class_weights)*self.loss_weights[1]

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
        self.iou_score2_tracker.update_state(K.flatten(multilabel_map),K.flatten(y_multilabel_resized))

        results = {m.name: m.result() for m in self.metrics}
        return results