

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers





def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(
        scale, mode="fan_avg", distribution="truncated_normal"
    )



def normalize(inputs):
    in_channels = inputs.shape[-1]
    if in_channels <= 16:
        num_groups = in_channels // 4
    else:
        num_groups = 16
    x = layers.GroupNormalization(groups=num_groups,epsilon=1e-5,)(inputs)
    return x

def ResidualBlock(width,  activation_fn=keras.activations.swish):
    def apply(inputs):
        x= inputs
        input_width = x.shape[-1]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(
                width, kernel_size=1, kernel_initializer=kernel_init(1.0)
            )(x)


        x = normalize(x)
        x = activation_fn(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0)
        )(x)

        x=normalize(x)
        x = activation_fn(x)

        x = layers.Conv2D(
            width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0)
        )(x)
        x = layers.Add()([x, residual])
        return x

    return apply



def DownSample(width,):
    def apply(x):
        x=tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x

    return apply



def UpSample(width,activation_fn, interpolation="nearest",):
    def apply(x):
        x = layers.Conv2DTranspose(
            width, kernel_size=5, padding="same", strides=(2,2),kernel_initializer=kernel_init(1.0)
        )(x)
        x=normalize(x)
        x=activation_fn(x)
        return x

    return apply
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
class ReScaler(keras.layers.Layer):
  def __init__(self, orig_shape):
    super().__init__()
    self.orig_shape=orig_shape
    
  def build(self,input_shape):
    size_n=int(input_shape[1])//int(self.orig_shape[1])
   # self.max_pooling=tf.keras.layers.MaxPooling2D(pool_size=(size_n,size_n))
   # self.windowspatching=WindowPatches(int(self.orig_shape[1]))
    self.proj_out =  layers.Conv2D(int(input_shape[-1]),kernel_size=1,padding="same", kernel_initializer=kernel_init(1.0))

  def call(self, inputs):
      x = tf.image.resize(inputs, (self.orig_shape[1],self.orig_shape[2]), method="bilinear")
      return self.proj_out(x)


def gelu(x):
    tanh_res = keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))
    return 0.5 * x * (1 + tanh_res)


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)

class GEGLU(keras.layers.Layer):
    def __init__(self, dim_out):
        super().__init__()
        self.proj = layers.Conv2D(dim_out*2,kernel_size=1,padding="same", kernel_initializer=kernel_init(1.0))
        self.dim_out = dim_out

    def call(self, x):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
        return x * quick_gelu(gate)


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim,):
        super().__init__()
        self.attn1 = AttentionBlock(dim)
        self.attn2 = CrossAttentionBlock(dim)
        self.geglu = GEGLU(dim * 4)
        self.dense  =layers.Conv2D(dim,kernel_size=1,padding="same", kernel_initializer=kernel_init(1.0))

    def call(self, inputs):
        x, context = inputs
        x = self.attn1(x) + x
        x = self.attn2([x, context]) + x
        return self.dense(self.geglu(x)) + x


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = layers.GroupNormalization(groups=16,epsilon=1e-5)
        self.proj_in =  layers.Conv2D(channels,kernel_size=1,padding="same", kernel_initializer=kernel_init(1.0))
        self.transformer_blocks = [BasicTransformerBlock(channels)]
        self.proj_out = layers.Conv2D(channels,kernel_size=1,padding="same", kernel_initializer=kernel_init(1.0))

    def build(self,input_shape):
        shape=input_shape[0]
        self.scaler=ReScaler(orig_shape=shape)


    def call(self, inputs):
        x, context = inputs
        context = x if context is None else context
        if context is not None:
          context=self.scaler(context)
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block([x, context])
        return self.proj_out(x) + x_in

class CrossAttentionBlock(layers.Layer):
    """Applies cross-attention.

    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)
    
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj =layers.Dense(units,kernel_initializer=kernel_init(0))

    def build(self,input_shape):
        in_channels=input_shape[0][-1]
        if in_channels <= 16:
            groups = in_channels // 4
        else:
            groups = 16
        self.norm = layers.GroupNormalization(groups=groups)
        
    def call(self, inputs):
        inputs, context = inputs
        context = inputs if context is None else context

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(context)
        v = self.value(context)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj

def vit(inputs,patch_size,num_patches,transformer_layers,projection_dim,num_heads,transformer_units,mlp_head_units):
    # Augment data.
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
         
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    features = mlp(representation, hidden_units=[x/2 for x in mlp_head_units], dropout_rate=0.5)


    return features

class AttentionBlock(layers.Layer):
    """Applies self-attention.
    Args:
        units: Number of units in the dense layers
        groups: Number of groups to be used for GroupNormalization layer
    """

    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)

        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj =layers.Dense(units,kernel_initializer=kernel_init(0))

    def build(self,input_shape):
        in_channels=input_shape[-1]
        if in_channels <= 16:
            groups = in_channels // 4
        else:
            groups = 16
        self.norm = layers.GroupNormalization(groups=groups)

        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])

        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])

        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj
    





import tensorflow as tf




class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.learned_skip = False
        self.batch_norm1 = tf.keras.layers.BatchNormalization( )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
            self.batch_norm3 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor: tf.Tensor):
        x = self.batch_norm1(input_tensor)
        x = self.conv_1(tf.nn.relu(x))
        x = self.batch_norm2(x)
        x = self.conv_2(tf.nn.relu(x))
        skip = (
            self.conv_3(tf.nn.relu(self.batch_norm3(input_tensor)))
            if self.learned_skip
            else input_tensor
        )
        output = skip + x

        return output




class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2DTranspose(self.filters , kernel_size=5, padding="same", strides=(2,2), kernel_initializer = 'he_normal')
    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        return x
    



"""

Get Decoder Weights Unet
"""


class UNet(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.unet_hidden_sizes=[int(x) for x in config.hidden_sizes]
        self.unet_hidden_sizes.insert(0,config.hidden_sizes[0])
        self.unet_hidden_sizes.insert(0,config.hidden_sizes[0]//2)

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("sigmoid")

        self.conv_first=tf.keras.layers.Conv2D(self.config.hidden_sizes[0]//2, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.config.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ResBlock(hidden_size)
                    self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

       

        self.middle_blocks=[ResBlock(self.unet_hidden_sizes[-1]),
                            ResBlock(self.unet_hidden_sizes[-1])
        ]

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.config.unet_num_res_blocks):
                    x = ResBlock(hidden_size)
                    self.decoder_blocks.append(x)
                
                if i!=0:

                   x = UpSample(hidden_size)
                   self.decoder_blocks.append(x)

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.final_layer=tf.keras.layers.Conv2D(1, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor):
        x=self.conv_first(input_tensor)
        skips=[x]
        hidden_states=[]
        for block in self.encoder_blocks:
            x=block(x)
            skips.append(x)
        for block in self.middle_blocks:
            x=block(x)

        for idx,block in enumerate(self.decoder_blocks):
            if idx in self.concat_idx[:-1]:
                  hidden_states.append(x)
       
            x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.batch_norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        hidden_states.reverse()
        return x,hidden_states
    
class SPADE(layers.Layer):
    def __init__(self, filters, epsilon=1e-5, **kwargs):
    
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.filters=filters
        self.conv = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv_gamma = layers.Conv2D(filters, 3, padding="same")
        self.conv_beta = layers.Conv2D(filters, 3, padding="same")

    def build(self, input_shape):

        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        mask = tf.image.resize(raw_mask, (self.resize_shape), method="nearest")
        x = self.conv(mask)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        output = ((gamma) * normalized) + beta
        return output
    

class PatchExtract(tf.keras.layers.Layer):

    
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]
    
    def call(self, images):
        
        batch_size = tf.shape(images)[0]
        
        patches = tf.image.extract_patches(images=images,
                                  sizes=(1, self.patch_size_x, self.patch_size_y, 1),
                                  strides=(1, self.patch_size_x, self.patch_size_y, 1),
                                  rates=(1, 1, 1, 1), padding='VALID',)
        # patches.shape = (num_sample, patch_num, patch_num, patch_size*channel)
        
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patches = tf.reshape(patches, (batch_size, patch_num*patch_num, patch_dim))
        # patches.shape = (num_sample, patch_num*patch_num, patch_size*channel)
        
        return patches
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size,})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class PatchEmbedding(tf.keras.layers.Layer):

    
    def __init__(self, num_patch, embed_dim, **kwargs):
        
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.proj = tf.keras.layers.Dense(embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patch': self.num_patch,
            'embed_dim': self.embed_dim,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        embed = self.proj(patch) + self.pos_embed(pos)
        return embed
    

class Gelu(tf.keras.layers.Layer):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
    def call(self,inputs):
        cdf = 0.5 * (1.0 + tf.math.erf(inputs / tf.cast(tf.sqrt(2.0), inputs.dtype)))
        return inputs * cdf


class MLP(tf.keras.layers.Layer):

    def __init__(self, filter_num, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.filter_num=filter_num
    def build(self,input_shape):
        self.MLP=[]
        for i, f in enumerate(self.filter_num):
           self.MLP.append(tf.keras.layers.Dense(f))
           self.MLP.append(Gelu())

    def call(self,inputs):
        x=inputs
        for block in self.MLP:
            x=block(x)
        return x



class VIT_block(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, filter_num_MLP, **kwargs):
        super(VIT_block, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.key_dim=key_dim
        self.filter_num_MLP=filter_num_MLP

    def build(self,input_shape):
        self.norm1=tf.keras.layers.LayerNormalization()
        self.multi_att=tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.norm2=tf.keras.layers.LayerNormalization()
        self.mlp=MLP(self.filter_num_MLP)
    def call(self,inputs):
        V_atten = self.norm1(inputs)
        V_atten = self.multi_att(V_atten)
        V_add = (V_atten+inputs)
        V_MLP = V_add 
        V_MLP = self.norm2(V_MLP)
        V_MLP = self.mlp(V_MLP)
        return V_MLP+V_add

class VIT(tf.keras.layers.Layer):
    def __init__(self,filter,patch_size,num_patches,embed_dim, num_transformer,num_heads, key_dim, filter_num_MLP,encode_size, **kwargs):
        super(VIT, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.key_dim=key_dim
        self.filter_num_MLP=filter_num_MLP
        self.num_transformer=num_transformer
        self.num_patches=num_patches
        self.patch_size=patch_size
        self.num_patches=num_patches
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.filter=filter
        self.encode_size=encode_size

    def build(self,input_shape):

        self.blocks=[]
        for i in range(len(self.num_transformer)):
            self.blocks.append(VIT_block(self.num_heads, self.key_dim, self.filter_num_MLP))


        self.conv_first=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')
        self.conv_final=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same",name="local_map", kernel_initializer = 'he_normal')

        self.patch_ext=PatchExtract((self.patch_size, self.patch_size))
        self.patch_emb=PatchEmbedding(self.num_patches, self.embed_dim)
        
    def call(self,inputs):
        x = self.conv_first(inputs)
        x = self.patch_extract(x)
        x = self.patch_embedding(x)
        for block in self.blocks:
            x=block(x)
        
        x=tf.reshape(x,(-1,self.encode_size,self.encode_size,self.embed_dim))

        x=self.conv_final(x)
        return x
    
