
from data.dataloader import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model.layers import *

def build_unet_model(
    input_shape,
    context_shape,
    channel_multiplier,
    has_attention,
    patch_size,
    num_patches,
    transformer_layers,
    projection_dim,
    num_heads,
    transformer_units,
    mlp_head_units,
    num_res_blocks=2,
    activation_fn=keras.activations.swish,
    first_conv_channels=64,
    use_spatial_transformer=True,
):


    widths = [first_conv_channels * mult for mult in channel_multiplier]
    pre_disaster_image= layers.Input(
        shape=context_shape, name="pre_img"
    )

    post_disaster_image = layers.Input(
        shape=input_shape, name="post_img"
    )

    context=layers.Conv2D(
        first_conv_channels,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(pre_disaster_image)

    x=layers.Concatenate()([pre_disaster_image,post_disaster_image])
    x = layers.Conv2D(
        first_conv_channels,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_init(1.0),
    )(x)


    skips = [x]

    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(
                widths[i], activation_fn=activation_fn
            )(x)
            if has_attention[i]:
                if use_spatial_transformer==True:
                    x = SpatialTransformer(widths[i])([x,context])
                else:
                    context_x=ReScaler(orig_shape=tf.shape(x))(context)
                    x = CrossAttentionBlock(widths[i])([x,context_x])

            skips.append(x)

        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)

    # MiddleBlock
    x = ResidualBlock(widths[-1] ,activation_fn=activation_fn)(
        x
    )
    if np.sum(np.asarray(has_attention)):
        if use_spatial_transformer==True:
            x = SpatialTransformer(widths[-1])([x,context])
        else:
            context_x=ReScaler(orig_shape=tf.shape(x))(context)
            x = CrossAttentionBlock(widths[-1])([x,context_x])

    x = ResidualBlock(widths[-1], activation_fn=activation_fn)(x)
       

    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(
                widths[i],activation_fn=activation_fn
            )(x)
            if has_attention[i]:
                if use_spatial_transformer==True:
                    x = SpatialTransformer(widths[i])([x,context])
                else:
                    context_x=ReScaler(orig_shape=tf.shape(x))(context)
                    x = CrossAttentionBlock(widths[i])([x,context_x])

        if i != 0:
            x = UpSample(widths[i],activation_fn=activation_fn)(x)

    # End block
    
    x = normalize(x)
    x = activation_fn(x)
    local_output = layers.Conv2D(1, 1, padding="same", kernel_initializer=kernel_init(0.0),activation="sigmoid")(x)

    y=layers.Concatenate()([local_output,pre_disaster_image,post_disaster_image])
    y=vit(y,patch_size,num_patches,transformer_layers,projection_dim,num_heads,transformer_units,mlp_head_units)
    # Create a [batch_size, projection_dim] tensor.
    y=layers.Reshape((128, 128,4))(y)
    y = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(y)
    y=layers.multiply([y,local_output])

    y = ResidualBlock(
                    16,activation_fn=activation_fn
                )(y)
    multi_output = layers.Conv2D(5, 1, padding="same", kernel_initializer=kernel_init(0.0),activation="sigmoid")(y)
    return keras.Model([pre_disaster_image,post_disaster_image ], [local_output,multi_output], name="spatial_condition_unet")



