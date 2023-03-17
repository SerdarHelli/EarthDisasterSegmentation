
def build_usegformernet_model(
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


    image_input = layers.Input(
        shape=input_shape, name="image_post"
    )
    image_input_pre= layers.Input(
        shape=context_shape, name="image_pre"
    )

    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=3,
        padding="same",
    )(image_input)

    context = layers.Conv2D(
        first_conv_channels,
        kernel_size=1,
        padding="same",
    )(image_input_pre)


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
                        patch_size=patch_sizes[i],
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
                        patch_size=patch_sizes[-1],
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
                        patch_size=patch_sizes[i],
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
    x = layers.Conv2D(5, (1, 1), padding="same")(x)
    x = tf.keras.layers.Activation("softmax")(x)

    return keras.Model([image_input, image_input_pre], x, name="usegformer_net")






class UnetSpatial_AutoEncoder(tf.keras.layers.Layer):

    """
        U-Net AutoEncoder:
        All blocks are resblock. 
        
        Between encoder and decoder , there is vit block.
        
        Paper Ref:
        TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
        Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou
    """
    def __init__(self,hidden_sizes,unet_num_res_blocks,unet_num_transformer,unet_num_heads,drop_path_rate,depths, **kwargs):
        super().__init__(**kwargs)
        self.hidden_sizes = hidden_sizes
        self.unet_num_res_blocks = unet_num_res_blocks
        self.unet_num_transformer=unet_num_transformer
        self.unet_num_heads=unet_num_heads
        self.unet_hidden_sizes=hidden_sizes
        self.norm="batchnorm"
        self.drop_path_rate=drop_path_rate

    def build(self,input_shape):
        self.final_activation = tf.keras.layers.Activation("softmax")

        self.conv_first=tf.keras.layers.Conv2D(self.hidden_sizes[0]//4, kernel_size=3,padding="same", kernel_initializer = 'he_normal')
        self.encoder_blocks=[]
        self.concat_idx=[]
        self.decoder_blocks=[]
        self.hidden_states_idx=[]
        idx_x=0
        for i,hidden_size in enumerate(self.unet_hidden_sizes):
                for _ in range(self.unet_num_res_blocks):
                    idx_x=idx_x+1
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.encoder_blocks.append(x)

                if hidden_size in self.unet_hidden_sizes[3:]:
                  idx_x=idx_x+1
                  x=SpatialTransformer(hidden_size//4)
                  self.encoder_blocks.append(x)

                if hidden_size != self.unet_hidden_sizes[-1]:
                  self.concat_idx.append(idx_x-1)
                  idx_x=idx_x+1
                  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                  self.encoder_blocks.append(x)

        self.middle_block=SpatialTransformer(self.unet_hidden_sizes[-1]//4)

        for i,hidden_size in reversed(list(enumerate(self.unet_hidden_sizes))):
                for _ in range(self.unet_num_res_blocks):
                    x = ResBlock(hidden_size,norm=self.norm,drop_path_rate=self.drop_path_rate)
                    self.decoder_blocks.append(x)
                if hidden_size in self.unet_hidden_sizes[3:]:
                  idx_x=idx_x+1
                  x=SpatialTransformer(hidden_size//4)
                  self.decoder_blocks.append(x)
                if i!=0:

                   x = UpSample(hidden_size,norm=self.norm)
                   self.decoder_blocks.append(x)

        self.norm = getNorm(self.norm)
        self.final_layer=tf.keras.layers.Conv2D(5, kernel_size=1,padding="same",name="classification_map", kernel_initializer = 'he_normal')

    def call(self, input_tensor: tf.Tensor,context_tensor:tf.Tensor,hidden_sizes):
        x=self.conv_first(input_tensor)
        skips=[x]
        curr=0
        for idx,block in enumerate(self.encoder_blocks):
            if idx in self.concat_idx[2:]:
                x = tf.concat([x, hidden_sizes[curr]], axis=-1)
                curr=curr+1
            if block.__class__.__name__=="SpatialTransformer":
                x=block([x,context_tensor])
            else:
                x=block(x)

            skips.append(x)

        x=self.middle_block([x,context_tensor])
        x = tf.concat([x, hidden_sizes[curr]], axis=-1)

        for idx,block in enumerate(self.decoder_blocks):
            if block.__class__.__name__=="SpatialTransformer":
                x=block([x,context_tensor])
            else:
                x=block(x)
            if (len(self.decoder_blocks)-1)-idx in self.concat_idx[1:]:
                  x = tf.concat([x, skips[(len(self.decoder_blocks)-1)-idx]], axis=-1)

        x=tf.nn.relu(self.norm(x))
        x=self.final_layer(x)
        x=self.final_activation(x)
        return x
    
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
        self.depthwise_convolution = tf.keras.layers.Conv2D(
            filters=in_features, kernel_size=3, strides=1, padding="same", groups=in_features, name="dwconv"
        )

        self.intermediate_act_fn = Gelu()

        self.dense2 = tf.keras.layers.Dense(out_features, name="dense2")

    def call(self, hidden_states: tf.Tensor,  training: bool = True) :
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.depthwise_convolution(hidden_states,)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states