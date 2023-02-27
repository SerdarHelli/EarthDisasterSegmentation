
import tensorflow as tf
from model.segformer import *

def getNorm(norm_str,eps=1e-6):
    x=None
    if norm_str=="batchnorm":
        x= tf.keras.layers.BatchNormalization(epsilon=eps)
    if norm_str=="layernorm":
        x= tf.keras.layers.LayerNormalization(epsilon=eps)
    if not x:
        raise("Invalid Normalization ")
    return x

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
        self.norm2 = getNorm(self.norm_str)


    def call(self, input_tensor: tf.Tensor):
        x = self.norm1(input_tensor)
        x = self.conv_1(tf.nn.relu(x))
        x = self.norm2(x)
        x = self.conv_2(tf.nn.relu(x))
        return x
    

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.learned_skip = False
        self.norm1 = getNorm(self.norm_str)
        self.norm2 = getNorm(self.norm_str)

        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
            self.norm3 = getNorm(self.norm_str)
        

    def call(self, input_tensor: tf.Tensor):
        x = self.norm1(input_tensor)
        x = self.conv_1(tf.nn.relu(x))
        x = self.norm2(x)
        x = self.conv_2(tf.nn.relu(x))
        skip = (
            self.conv_3(tf.nn.relu(self.norm3(input_tensor)))
            if self.learned_skip
            else input_tensor
        )
        output = skip + x

        return output
    
class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2DTranspose(self.filters , kernel_size=5, padding="same", strides=(2,2), kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        return self.norm1(x)

class DownSample(tf.keras.layers.Layer):
    def __init__(self, filters,norm="layernorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, kernel_size=2, strides=2,padding="same", kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        
        return self.norm1(x)

class ConvNeXtBlock(tf.keras.layers.Layer):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6,norm="layernorm", prefix=''):
        super().__init__()
        self.norm_str=norm
        self.dwconv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=7, padding='same')  # depthwise conv
        self.norm1 = getNorm(self.norm_str)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = tf.keras.layers.Dense(4 * dim)
        self.act = Gelu()
        self.pwconv2 = tf.keras.layers.Dense(dim)
        self.drop_path = TFSegformerDropPath(drop_path)
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value
        self.prefix = prefix
        self.norm2 = getNorm(self.norm_str)

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.learned_skip = False

        self.gamma = tf.Variable(
            initial_value=self.layer_scale_init_value * tf.ones((self.dim)),
            trainable=True,
            name=f'{self.prefix}/gamma')
        self.built = True

        if self.dim != input_filter:
            self.learned_skip = True
            self.pwconv_skip = tf.keras.layers.Dense(self.dim)
            self.norm_skip = getNorm(self.norm_str)
            self.act_skip = Gelu()

    def call(self, inputs: tf.Tensor):
        x=self.norm1(inputs)
        x = self.dwconv(tf.nn.relu(x))
        # x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        skip = (
                self.pwconv_skip(self.act_skip(self.norm_skip(inputs)))
                if self.learned_skip
                else inputs
            )
        x = skip + self.drop_path(x)
        return x