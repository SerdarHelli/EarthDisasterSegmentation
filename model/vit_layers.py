import tensorflow as tf

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
        V_atten = self.multi_att(V_atten,V_atten)
        V_add = (V_atten+inputs)
        V_MLP = V_add 
        V_MLP = self.norm2(V_MLP)
        V_MLP = self.mlp(V_MLP)
        return V_MLP+V_add

class VIT(tf.keras.layers.Layer):
    def __init__(self,filter,embed_dim, num_transformer,num_heads,patch_size=1, **kwargs):
        super(VIT, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.num_transformer=num_transformer
        self.filter_num_MLP=[embed_dim//2,embed_dim]
        self.key_dim=embed_dim
        #its constant
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.filter=filter

    def build(self,input_shape):
        self.encode_size=int(input_shape[1])
        self.num_patches=(int(input_shape[1])//self.patch_size)**2

        self.blocks=[]
        for i in range(self.num_transformer):
            self.blocks.append(VIT_block(self.num_heads, self.key_dim, self.filter_num_MLP))


        self.conv_first=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same", kernel_initializer = 'he_normal')
        self.conv_final=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same", kernel_initializer = 'he_normal')

        self.patch_ext=PatchExtract((self.patch_size, self.patch_size))
        self.patch_emb=PatchEmbedding(self.num_patches, self.embed_dim)
        
    def call(self,inputs):
        x = self.conv_first(inputs)
        x = self.patch_ext(x)
        x = self.patch_emb(x)
        for block in self.blocks:
            x=block(x)
        
        x=tf.reshape(x,(-1,self.encode_size,self.encode_size,self.embed_dim))

        x=self.conv_final(x)
        return x
    


class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int,**kwargs) :
        super(MultiHeadCrossAttention, self).__init__(**kwargs)
        self.embed_dim=embed_dim
        self.num_heads=num_heads


    def build(self,input_shape):

        self.conv_senc = []
        self.conv_senc.append(tf.keras.layers.Conv2D(self.embed_dim, kernel_size=1,padding="same", kernel_initializer = 'he_normal'))
        self.conv_senc.append(tf.keras.layers.BatchNormalization())
        self.conv_senc.append(tf.keras.layers.Activation("relu"))

        self.conv_S = []
        self.conv_S.append(tf.keras.layers.MaxPooling2D())
        self.conv_S.append(tf.keras.layers.Conv2D(self.embed_dim, kernel_size=1,padding="same", kernel_initializer = 'he_normal'))
        self.conv_S.append(tf.keras.layers.BatchNormalization())
        self.conv_S.append(tf.keras.layers.Activation("relu"))

        self.conv_Y = []
        self.conv_Y.append(tf.keras.layers.Conv2D(self.embed_dim, kernel_size=1,padding="same", kernel_initializer = 'he_normal'))
        self.conv_Y.append(tf.keras.layers.BatchNormalization())
        self.conv_Y.append(tf.keras.layers.Activation("relu"))
       

        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=self.embed_dim, num_heads=self.num_heads)

        self.upsample = []
        self.upsample.append(tf.keras.layers.Conv2D(self.embed_dim, kernel_size=1,padding="same", kernel_initializer = 'he_normal'))
        self.upsample.append(tf.keras.layers.BatchNormalization())
        self.upsample.append(tf.keras.layers.Activation("sigmoid"))
        self.upsample.append(tf.keras.layers.Conv2DTranspose(self.embed_dim, kernel_size=2,padding="same", strides=2,kernel_initializer = 'he_normal'))

        

    def call(self, input_tensor) :
        s,y=input_tensor
        s_enc = s
        for block in self.conv_senc:
            s_enc=block(s_enc)

        for block in self.conv_S:
            s=block(s)

        for block in self.conv_Y:
            y=block(y)

        s_o=tf.reshape(s,(tf.shape(s)[0],tf.shape(s)[1]*tf.shape(s)[2],self.embed_dim))
        y_o=tf.reshape(y,(tf.shape(y)[0],tf.shape(y)[1]*tf.shape(y)[2],self.embed_dim))

        x = self.mha(s_o,y_o)
        x=tf.reshape(x,(-1,tf.shape(s)[1],tf.shape(s)[2],self.embed_dim))

        for block in self.upsample:
            x=block(x)

        out=x*s_enc
        return out


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int,**kwargs) :
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim=embed_dim
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)


    def call(self, input_tensor) :
        s=input_tensor
        s=tf.reshape(s,(tf.shape(s)[0],tf.shape(s)[1]*tf.shape(s)[2],self.embed_dim))
        x = self.mha(s,s)
        x=tf.reshape(x,(tf.shape(input_tensor)[0],tf.shape(input_tensor)[1],tf.shape(input_tensor)[2],self.embed_dim))
        return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,**kwargs) :
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, x) :
        shape= tf.shape(x)
        b=int(shape[0])
        h=int(shape[1])
        w=int(shape[2])
        c=int(shape[3])

        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding=tf.tile(pos_encoding,[b,1])
        pos_encoding=tf.reshape(pos_encoding,[b,h * w, c])
        x=tf.reshape(x,(b,h*w,c))+pos_encoding
        return tf.reshape(x,(b,h,w,c))

    def positional_encoding(self, length: int, depth: int):
        depth = depth / 2

        positions = tf.cast(tf.range(start=0, limit=length, delta=1),dtype=tf.float32)
        depths = tf.range(start=0,limit=depth) / depth

        angle_rates = tf.cast(1 / (10000**depths),dtype=tf.float32)
        angle_rads = tf.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = tf.concat((tf.math.sin(angle_rads), tf.math.cos(angle_rads)), axis=-1)

        return pos_encoding


class MultiCrossAttention(tf.keras.layers.Layer):
    def __init__(self,embed_dim,out_embed_dim,num_heads,**kwargs):
        super(MultiCrossAttention, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.embed_dim=embed_dim
        self.out_embed_dim=out_embed_dim

    def build(self,input_shape):
        self.mhca=MultiHeadCrossAttention(embed_dim=self.embed_dim,num_heads=self.num_heads)
        self.positional_encoding_input=PositionalEncoding()
        self.positional_encoding_context=PositionalEncoding()
        self.x_out=tf.keras.layers.Conv2D(self.out_embed_dim, kernel_size=1,padding="same", kernel_initializer = 'he_normal')
        self.norm_out=tf.keras.layers.BatchNormalization()
        self.x_out2=tf.keras.layers.Conv2D(self.out_embed_dim, kernel_size=1,padding="same", kernel_initializer = 'he_normal')
        self.norm_out2=tf.keras.layers.BatchNormalization()

    def call(self,input_tensor):
        inputs,context=input_tensor
        x = self.positional_encoding_input(inputs)
        context = self.positional_encoding_context(context)

        skill=self.mhca([x,context])

        skill=self.x_out2(skill)
        skill=tf.nn.relu(self.norm_out2(skill))
        x=self.x_out(x)
        x=tf.nn.relu(self.norm_out(x))
        return x,skill