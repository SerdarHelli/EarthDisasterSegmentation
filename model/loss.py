import tensorflow.keras.backend as K
import tensorflow as tf 
import numpy as np


class JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1,**kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth

    def call(self, y_true, y_pred):
      intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
      union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
      iou = K.mean((intersection + self.smooth) / (union + self.smooth), axis=0)
      return (1-iou)

class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1,**kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth
        self.epsilon=K.epsilon()
        self.alpha=0.5
        self.beta=0.5
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice= K.mean( (2. * intersection + self.smooth) / (union + self.smooth), axis=0)

        y_pred = K.clip(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = self.bce(K.flatten(y_true),K.flatten(y_pred))
        beta_weight = np.array([self.beta, 1-self.beta])
        cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        combo_loss = (self.alpha * cross_entropy) - ((1 - self.alpha) * dice)
        return combo_loss



class GeneralizedDice(tf.keras.losses.Loss):
    def __init__(self, smooth=1,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=K.epsilon()


    def call(self, y_true, y_pred):
        y_true = K.clip(y_true, self.epsilon, 1. - self.epsilon)
        y_pred = K.clip(y_pred, self.epsilon, 1. - self.epsilon)

        weights = 1./K.sum(y_true, axis=[0,1,2])
        weights = weights/K.sum(weights)
        num = K.sum(weights*K.sum(y_true*y_pred, axis=[0,1,2]))
        den = K.sum(weights*K.sum(y_true+y_pred, axis=[0,1,2]))
        loss=K.mean(2.*(num+self.epsilon)/(den+self.epsilon))
        return loss
    
class FocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.smooth=0.000001
        self.epsilon=K.epsilon()
        self.delta=0.7
        self.gamma=0.75

    def call(self, y_true, y_pred):
        y_pred = K.clip(y_pred, self.epsilon, 1. - self.epsilon) 

        tp = K.sum(y_true * y_pred, axis=[1,2,3])
        fn = K.sum(y_true * (1-y_pred), axis=[1,2,3])
        fp = K.sum((1-y_true) * y_pred, axis=[1,2,3])
        tversky_class = (tp + self.smooth)/(tp + self.delta*fn + (1-self.delta)*fp +  self.smooth)
        # Average class scores
        focal_tversky_loss = K.mean(K.pow((1-tversky_class), self.gamma))
        return focal_tversky_loss

