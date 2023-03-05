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
        self.num_patches=int(input_shape[1])**2

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
    

class VIT_block_Cross(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, filter_num_MLP, **kwargs):
        super(VIT_block_Cross, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.key_dim=key_dim
        self.filter_num_MLP=filter_num_MLP

    def build(self,input_shape):
        self.norm1=tf.keras.layers.LayerNormalization()
        self.multi_att=tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.norm2=tf.keras.layers.LayerNormalization()
        self.mlp=MLP(self.filter_num_MLP)
    def call(self,inputs,context):
        V_atten = self.norm1(inputs)
        V_atten = self.multi_att(V_atten,context)
        V_add = (V_atten+inputs)
        V_MLP = V_add 
        V_MLP = self.norm2(V_MLP)
        V_MLP = self.mlp(V_MLP)
        return V_MLP+V_add  
    

class VITCross(tf.keras.layers.Layer):
    def __init__(self,filter,embed_dim, num_transformer,num_heads,patch_size=1, **kwargs):
        super(VITCross, self).__init__(**kwargs)
        self.num_heads=num_heads
        self.num_transformer=num_transformer
        self.filter_num_MLP=[embed_dim//2,embed_dim]
        self.key_dim=embed_dim
        #its constant
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        self.filter=filter
        
    def build(self,input_shape):
        self.encode_size=int(input_shape[0][1])
        self.num_patches=int(input_shape[0][1])**2
        self.blocks=[]
        for i in range(self.num_transformer):
            self.blocks.append(VIT_block_Cross(self.num_heads, self.key_dim, self.filter_num_MLP))


        self.conv_first=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same", kernel_initializer = 'he_normal')
        self.conv_final=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same", kernel_initializer = 'he_normal')

        self.patch_ext=PatchExtract((self.patch_size, self.patch_size))
        self.patch_emb=PatchEmbedding(self.num_patches, self.embed_dim)

        self.conv_first_context=tf.keras.layers.Conv2D(self.filter, kernel_size=1,padding="same", kernel_initializer = 'he_normal')
        self.patch_ext_context=PatchExtract((self.patch_size, self.patch_size))
        self.patch_emb_context=PatchEmbedding(self.num_patches, self.embed_dim)

    def call(self,input_tensor):
        inputs,context=input_tensor
        x = self.conv_first(inputs)
        x = self.patch_ext(x)
        x = self.patch_emb(x)
        y = self.conv_first_context(context)
        y = self.patch_ext_context(y)
        y = self.patch_emb_context(y)

        for block in self.blocks:
            x=block(x,y)
        
        x=tf.reshape(x,(-1,self.encode_size,self.encode_size,self.embed_dim))

        x=self.conv_final(x)
        return x
    