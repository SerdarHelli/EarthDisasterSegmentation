
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
    


class SEResBlock(tf.keras.layers.Layer):
    #ResNext
    def __init__(self, filters,cardinality,drop_path_rate=0,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
        self.cardinality=cardinality
        self.drop_path_rate=drop_path_rate

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.grouped_channels=int(self.filters//self.cardinality)

        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 1, padding="same", use_bias=False,kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')

        self.learned_skip = False
        self.norm0 = getNorm(self.norm_str)
        self.norm1 = getNorm(self.norm_str)
        self.norm2 = getNorm(self.norm_str)
        self.norm3 = getNorm(self.norm_str)
        self.droput=tf.keras.layers.Dropout(self.drop_path_rate)

        self.se= SqueezeAndExcite2D(filters=self.filters)
        if self.filters != input_filter:
            self.learned_skip = True
            self.conv_3 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
            self.norm4 = getNorm(self.norm_str)

        self.grouped_convs=[]
        for c in range(self.cardinality):
            self.grouped_convs.append(tf.keras.layers.Conv2D(self.grouped_channels, (3, 3), padding='same', use_bias=False,
                                                             kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(1e-6)))

            
    def call(self, input_tensor: tf.Tensor):
        x = tf.nn.relu(self.norm0(input_tensor))
        x=self.conv_1(x)
        x = tf.nn.relu(self.norm1(x))


        groups=[]
        for idx,block in enumerate(self.grouped_convs):
            groups.append(block(x[:, :, :, idx * self.grouped_channels:(idx + 1) * self.grouped_channels]))

        x=tf.concat(groups, axis=-1)
        x = tf.nn.relu(self.norm2(x))
        x=self.se(x)
        x = self.conv_2(x)
        x = tf.nn.relu(self.norm3(x))

        skip = (
            self.conv_3(tf.nn.relu(self.norm4(input_tensor)))
            if self.learned_skip
            else input_tensor
        )
        
        output = skip + x
        output=self.droput(output)

        return output



class AttentionGate(tf.keras.layers.Layer):
    def __init__(self, filters,do_upsample=True,**kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.do_upsample=do_upsample
    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")
        self.phi_g = tf.keras.layers.Conv2D(self.filters, 1, padding="same", kernel_initializer = 'he_normal')
        self.theta_att = tf.keras.layers.Conv2D(self.filters, 1, padding="same", kernel_initializer = 'he_normal')
        self.psi_f = tf.keras.layers.Conv2D(1, 1, padding="same", kernel_initializer = 'he_normal')
        self.f=tf.keras.layers.Activation("relu")
        self.coef_att=tf.keras.layers.Activation("sigmoid")

    def call(self, gated_tensor: tf.Tensor,input_tensor:tf.Tensor):
        if self.do_upsample==True:
            gated_tensor=self.upsample(gated_tensor)
        theta_att = self.theta_att(input_tensor)
        phi_g = self.phi_g(gated_tensor)
        query=theta_att+phi_g
        f=self.f(query)
        psi_f = self.psi_f(f)
        coef_att = self.coef_att(psi_f)
        X_att=coef_att*input_tensor
        return X_att

class USENETBlock(tf.keras.layers.Layer):
    def __init__(self, filters,**kwargs):
        super().__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation="bilinear")
        self.conv1=tf.keras.layers.Conv2D(self.filters, 1, padding="same", kernel_initializer = 'he_normal')
        self.conv2=tf.keras.layers.Conv2D(self.filters, 1, padding="same", kernel_initializer = 'he_normal')
        self.se_skip=SqueezeAndExcite2D(self.filters)
        self.se_decoder=SqueezeAndExcite2D(self.filters)

    def call(self, input_tensor: tf.Tensor,skip_tensor:tf.Tensor):
        input_tensor=self.conv1(input_tensor)
        skip_tensor=self.conv2(skip_tensor)
        x1=self.upsample(input_tensor)
        x1=self.se_decoder(x1)
        x1=x1+input_tensor
        x2 = self.se_skip(skip_tensor)
        x2=x2+skip_tensor
        x=x1+x2
        return x
     

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
    
class ConvSEBlock(tf.keras.layers.Layer):
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
        self.act1=tf.keras.layers.Activation("relu")
        self.act2=tf.keras.layers.Activation("relu")
        self.se= SqueezeAndExcite2D(filters=self.filters)
        self.conv_3 = tf.keras.layers.Conv2D(self.filters, 1, padding="same", kernel_initializer = 'he_normal')
        self.norm3 = getNorm(self.norm_str)
        self.act3=tf.keras.layers.Activation("relu")
        self.droput=tf.keras.layers.Dropout(self.drop_path_rate)

    def call(self, input_tensor: tf.Tensor):
        x = self.norm1(input_tensor)
        x=self.act1(x)
        x = self.conv_1(x)
        x = self.norm2(x)
        x=self.act2(x)
        x = self.conv_2(x)
        x=self.norm3(x)
        x=self.act3(x)
        x=self.se(x)
        x=self.conv_3(x)
        x=self.droput(x)
        return x
    
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters,drop_path_rate=0,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
        self.drop_path_rate=drop_path_rate
    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.conv_2 = tf.keras.layers.Conv2D(self.filters, 3, padding="same", kernel_initializer = 'he_normal')
        self.learned_skip = False
        self.norm1 = getNorm(self.norm_str)
        self.norm2 = getNorm(self.norm_str)
        self.droput=tf.keras.layers.Dropout(self.drop_path_rate)

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
        output=self.droput(output)
        return output
    
class UpSample(tf.keras.layers.Layer):
    def __init__(self, filters,norm="batchnorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2DTranspose(self.filters , kernel_size=5, padding="same", strides=(2,2), kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
        self.act=tf.keras.layers.Activation("relu")

    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        x=self.norm1(x)
        x=self.act(x)
        return x

class DownSample(tf.keras.layers.Layer):
    def __init__(self, filters,norm="layernorm", **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.norm_str=norm
    def build(self, input_shape):
        self.conv_1 = tf.keras.layers.Conv2D(self.filters, kernel_size=2, strides=2,padding="same", kernel_initializer = 'he_normal')
        self.norm1 = getNorm(self.norm_str)
        self.act=tf.keras.layers.Activation("relu")

    def call(self, input_tensor: tf.Tensor):
        x = self.conv_1(input_tensor)
        x=self.norm1(x)
        x=self.act(x)
        return x

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
    
