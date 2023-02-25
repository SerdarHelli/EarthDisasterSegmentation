
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