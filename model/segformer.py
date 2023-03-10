#ref Huggingg Face Segformer

import tensorflow as tf
import math
from typing import Optional

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
        self.padding = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.proj = tf.keras.layers.Conv2D(
            filters=hidden_size, kernel_size=patch_size, strides=stride, padding="VALID", name="proj"
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-05, name="layer_norm")

    def call(self, inputs):
        embeddings = self.proj(self.padding(inputs))
        shape=tf.shape(embeddings)

        height = shape[1]
        width = shape[2]
        hidden_dim = shape[3]
        # (batch_size, height, width, num_channels) -> (batch_size, height*width, num_channels)
        # this can be fed to a Transformer layer
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width

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
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = True,
    ) :
        batch_size = tf.shape(hidden_states)[0]
        num_channels = tf.shape(hidden_states)[2]

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            # Reshape to (batch_size, height, width, num_channels)
            hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))
            # Apply sequence reduction
            hidden_states = self.sr(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = tf.reshape(hidden_states, (batch_size, -1, num_channels))
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

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

    def call(self, hidden_states: tf.Tensor, training: bool = True) -> tf.Tensor:

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
        self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool = False
    ) :
        self_outputs = self.self(hidden_states, height, width, output_attentions)

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

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        decoder_hidden_size=config.decoder_hidden_size
        self.norm=tf.keras.layers.LayerNormalization()
        self.proj = tf.keras.layers.Dense(decoder_hidden_size, name="proj")

    def call(self, hidden_states: tf.Tensor):
        hidden_states=self.norm(hidden_states)
        hidden_states = self.proj(hidden_states)
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

    def call(
        self,
        hidden_states: tf.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
        training: bool = True,
    ) :
        self_attention_outputs = self.attention(
            self.layer_norm_1(hidden_states),  # in Segformer, layernorm is applied before self-attention
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

        outputs = (layer_output,) + outputs

        return outputs
    


class TFSegformerEncoder(tf.keras.Model):
    def __init__(self, config,**kwargs):
        super().__init__(**kwargs)
        self.config=config


        # stochastic depth decay rule
        drop_path_decays = [x.numpy() for x in tf.linspace(0.0, 0.1, sum(config.depths))]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                TFSegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    hidden_size=config.hidden_sizes[i],
                    name=f"patch_embeddings.{i}",
                )
            )
        self.embeddings = embeddings

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    TFSegformerLayer(
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                        name=f"block.{i}.{j}"
                    )
                )
            blocks.append(layers)

        self.block = blocks

        # Layer norms
        self.layer_norms = [
            tf.keras.layers.LayerNormalization(epsilon=1e-05, name=f"layer_norm.{i}")
            for i in range(config.num_encoder_blocks)
        ]

    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        training: bool = True,
    ) :
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = tf.shape(pixel_values)[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.embeddings, self.block, self.layer_norms)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)

            # second, send embeddings through blocks
            # (each block consists of multiple layers i.e., list of layers)
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(
                    hidden_states,
                    height,
                    width,
                    output_attentions,
                    training=training,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)

            # fourth, optionally reshape back to (batch_size, height, width, num_channels)
            if idx != len(self.embeddings) - 1 or (idx == len(self.embeddings) - 1 and self.config.reshape_last_stage):
                num_channels = tf.shape(hidden_states)[-1]
                hidden_states = tf.reshape(hidden_states, (batch_size, height, width, num_channels))

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        return hidden_states, all_hidden_states, all_self_attentions


class TFSegformerMainLayer(tf.keras.Model):

    def __init__(self,config,  **kwargs):
        super().__init__(**kwargs)

        # hierarchical Transformer encoder
        self.encoder = TFSegformerEncoder(config, name="encoder")
        self.config=config

    def call(
        self,
        pixel_values: tf.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = True,
    ) :
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # When running on CPU, `tf.keras.layers.Conv2D` doesn't support `NCHW` format.
        # So change the input format from `NCHW` to `NHWC`.
        # shape = (batch_size, in_height, in_width, in_channels=num_channels)

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = encoder_outputs[0]
        # Change to NCHW output format to have uniformity in the modules
        sequence_output = tf.transpose(sequence_output, perm=[0, 3, 1, 2])

        # Change the other hidden state outputs to NCHW as well
        if output_hidden_states:
            hidden_states = tuple([tf.transpose(h, perm=(0, 3, 1, 2)) for h in encoder_outputs[1]])

        if not return_dict:
            if tf.greater(len(encoder_outputs[1:]), 0):
                transposed_encoder_outputs = tuple(tf.transpose(v, perm=[0, 3, 1, 2]) for v in encoder_outputs[1:][0])
                return (sequence_output,) + (transposed_encoder_outputs,)
            else:
                return (sequence_output,) + encoder_outputs[1:]
        hidden_states=hidden_states if output_hidden_states else encoder_outputs[1]
        return sequence_output,hidden_states,encoder_outputs[-1]
          

class TFSegformerDecodeHead(tf.keras.Model):
    def __init__(self,config, **kwargs):
        super().__init__( **kwargs)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        self.config=config
        mlps = []
        unet_mlps=[]
        for i in range(config.num_encoder_blocks):
            mlp = TFSegformerMLP(config, name=f"linear_c.{i}")
            mlps.append(mlp)
            unet_mlp = TFSegformerMLP(config, name=f"unet_linear_c.{i}")
            unet_mlps.append(unet_mlp)
        self.mlps = mlps
        self.unet_mlps = unet_mlps

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = tf.keras.layers.Conv2D(
            filters=config.decoder_hidden_size, kernel_size=1, use_bias=False, name="linear_fuse",activity_regularizer=tf.keras.regularizers.L2(1e-6)
        )
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="batch_norm")
        self.activation = tf.keras.layers.Activation("relu")

        self.dropout = tf.keras.layers.Dropout(config.classifier_dropout_prob)



        self.classifier = tf.keras.layers.Conv2D(filters=config.num_labels, kernel_size=1, name="classifier")


    def call(self, encoder_hidden_states,unet_hidden_states, training: bool = True):
        batch_size = tf.shape(encoder_hidden_states[-1])[0]

        all_hidden_states = ()
        for idx,(encoder_hidden_state, mlp,unet_mlp) in enumerate(zip(encoder_hidden_states, self.mlps,self.unet_mlps)):
            if self.config.reshape_last_stage is False and len(tf.shape(encoder_hidden_state)) == 3:
                height = tf.math.sqrt(tf.cast(tf.shape(encoder_hidden_state)[1], tf.float32))
                height = width = tf.cast(height, tf.int32)
                encoder_hidden_state = tf.reshape(encoder_hidden_state, (batch_size, height, width, -1))

            # unify channel dimension
            

            encoder_hidden_state = tf.transpose(encoder_hidden_state, perm=[0, 2, 3, 1])
            height = tf.shape(encoder_hidden_state)[1]
            width = tf.shape(encoder_hidden_state)[2]
            encoder_hidden_state = mlp(encoder_hidden_state)
            unet_hidden_state = unet_mlp(unet_hidden_states[idx])
            encoder_hidden_state=tf.concat([encoder_hidden_state,unet_hidden_state],axis=-1)

            # upsample
            temp_state = tf.transpose(encoder_hidden_states[0], perm=[0, 2, 3, 1])
            upsample_resolution = tf.shape(temp_state)[1:-1]
            
            encoder_hidden_state = tf.image.resize(encoder_hidden_state, size=upsample_resolution, method="bilinear")
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(tf.concat(all_hidden_states[::-1], axis=-1))
        hidden_states = self.batch_norm(hidden_states, training=training)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
    
        # logits of shape (batch_size, height/4, width/4, num_labels)
        logits = self.classifier(hidden_states)

        return logits
    
class TFSegformerForSemanticSegmentation(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__( **kwargs)
        self.segformer = TFSegformerMainLayer(config, name="segformer")
        self.decode_head = TFSegformerDecodeHead(config, name="decode_head")
        self.final_activation = tf.keras.layers.Activation("sigmoid")
        self.config=config

    def call(
        self,
        pixel_values: tf.Tensor,
        unet_hidden_states,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) :

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs[1] 

        logits = self.decode_head(encoder_hidden_states,unet_hidden_states)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return output

        return self.final_activation(logits)
    

