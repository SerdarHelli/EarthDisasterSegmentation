
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


class SqueezeAndExcite2D(tf.keras.layers.Layer):
    """
    ref : KerasCv ref https://github.com/keras-team/keras-cv/blob/v0.4.2/keras_cv/layers/regularization/squeeze_excite.py#L20
    Implements Squeeze and Excite block as in
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).
    This layer tries to use a content aware mechanism to assign channel-wise
    weights adaptively. It first squeezes the feature maps into a single value
    using global average pooling, which are then fed into two Conv1D layers,
    which act like fully-connected layers. The first layer reduces the
    dimensionality of the feature maps by a factor of `ratio`, whereas the second
    layer restores it to its original value.
    The resultant values are the adaptive weights for each channel. These
    weights are then multiplied with the original inputs to scale the outputs
    based on their individual weightages.
    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        ratio: Ratio for bottleneck filters. Number of bottleneck filters =
            filters * ratio. Defaults to 0.25.
        squeeze_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after squeeze convolution. Defaults to `relu`.
        excite_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after excite convolution. Defaults to `sigmoid`.
    Usage:
    ```python
    # (...)
    input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(16, (3, 3))(input)
    output = keras_cv.layers.SqueezeAndExciteBlock(16)(x)
    # (...)
    ```
    """

    def __init__(
        self,
        filters,
        ratio=0.25,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters

        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"`ratio` should be a float between 0 and 1. Got {ratio}")

        if filters <= 0 or not isinstance(filters, int):
            raise ValueError(f"`filters` should be a positive integer. Got {filters}")

        self.ratio = ratio
        self.bottleneck_filters = int(self.filters * self.ratio)

        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation

        self.global_average_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.squeeze_conv = tf.keras.layers.Conv2D(
            self.bottleneck_filters,
            (1, 1),
            activation=self.squeeze_activation,padding="same", kernel_initializer = 'he_normal'
        )
        self.excite_conv = tf.keras.layers.Conv2D(
            self.filters, (1, 1), activation=self.excite_activation,padding="same", kernel_initializer = 'he_normal'
        )

    def call(self, inputs, training=True):
        x = self.global_average_pool(inputs)  # x: (batch_size, 1, 1, filters)
        x = self.squeeze_conv(x)  # x: (batch_size, 1, 1, bottleneck_filters)
        x = self.excite_conv(x)  # x: (batch_size, 1, 1, filters)
        x = tf.math.multiply(x, inputs)  # x: (batch_size, h, w, filters)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "ratio": self.ratio,
            "squeeze_activation": self.squeeze_activation,
            "excite_activation": self.excite_activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
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
        self.dwconv = tf.keras.layers.Conv2D(filters=dim,groups=dim,
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
    

class SPADE(tf.keras.layers.Layer):
    def __init__(self, filters, epsilon=1e-5, **kwargs):
    
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.filters=filters
        self.conv = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")
        self.conv_gamma = tf.keras.layers.Conv2D(filters, 3, padding="same")
        self.conv_beta = tf.keras.layers.Conv2D(filters, 3, padding="same")

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