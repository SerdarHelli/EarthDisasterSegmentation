import tensorflow.keras.backend as K
import tensorflow as tf 
import numpy as np

M_tree_5 = np.array([[0., 1., 1., 1., 1.],
                   [1., 0., 0.6, 0.2, 0.5],
                   [1., 0.6, 0., 0.6, 0.7],
                   [1., 0.2, 0.6, 0., 0.5],
                   [1., 0.5, 0.7, 0.5, 0.]], dtype=np.float64)

M_tree_4 = np.array([[0., 1., 1., 1.,],
                     [1., 0., 0.6, 0.5],
                     [1., 0.6, 0., 0.7],
                     [1., 0.5, 0.7, 0.]], dtype=np.float64)



class JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1,**kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth

    def call(self, y_true, y_pred):
      intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
      union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
      iou = K.mean((intersection + self.smooth) / (union + self.smooth), axis=0)
      return (1-iou)

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2,**kwargs ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.epsilon=K.epsilon()

    def call(self, y_true, y_pred):
        y_pred = K.clip(y_pred, self.epsilon, 1. - self.epsilon)
        y_true = K.clip(y_true, self.epsilon, 1. - self.epsilon)
        pt = (1 - y_true) * (1 - y_pred) + y_true * y_pred
        return K.mean((-(1. - pt) ** self.gamma * K.log(pt)))

class DiceLoss(tf.keras.losses.Loss):
 
    def __init__(self,weight=None,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=K.epsilon()


    def call(self, y_true, y_pred):

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1])
        union = K.sum(y_true, axis=[1]) + K.sum(y_pred, axis=[1])
        dice= 1- ((2. * intersection + self.epsilon) / (union + self.epsilon))

        dice=K.mean(dice)
        return dice
    
    
class ComboLoss(tf.keras.losses.Loss):
    """
    ALPHA = 0.3    # < 0.5 penalises FP more, > 0.5 penalises FN more
    CE_RATIO = 0.7 # weighted contribution of modified CE loss compared to Dice loss
    """
    def __init__(self,alpha=0.3,ce_ratio=0.7, smooth=1,**kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth
        self.epsilon=K.epsilon()
        self.alpha=alpha
        self.ce_ratio=ce_ratio

    def call(self, y_true, y_pred):

        targets = K.flatten(y_true)
        inputs = K.flatten(y_pred)
        
        intersection = K.sum(targets * inputs)
        dice = (2. * intersection + self.smooth) / (K.sum(targets) + K.sum(inputs) + self.smooth)
        inputs = K.clip(inputs, self.epsilon, 1.0 - self.epsilon)
        out = - (self.alpha * ((targets * K.log(inputs)) + ((1 - self.alpha) * (1.0 - targets) * K.log(1.0 - inputs))))
        weighted_ce = K.mean(out, axis=-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        return combo
"""
    It is not working well

class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=0.0001,**kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth
        self.epsilon=K.epsilon()
        self.alpha=0.5
        self.beta=0.5
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice=1- K.mean( (2. * intersection + self.smooth) / (union + self.smooth), axis=0)

        y_pred = K.clip(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = self.bce(K.flatten(y_true),K.flatten(y_pred))
        if self.beta is not None:
            beta_weight = np.array([self.beta, 1-self.beta])
            cross_entropy = beta_weight * cross_entropy
        # sum over classes
        cross_entropy = K.mean(K.sum(cross_entropy, axis=[-1]))
        if self.alpha is not None:
            combo_loss = (self.alpha * cross_entropy) + ((1 - self.alpha) * dice)
        else:
            combo_loss = cross_entropy - dice
        return combo_loss
"""

class GeneralizedFocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self,gamma=4/3, smooth=0.000001, **kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth
        self.epsilon=K.epsilon()
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        y_true_shape = tf.shape(y_true)
        hwc_shape= y_true_shape[1]*y_true_shape[2]* y_true_shape[3]
        targets = tf.reshape(y_true, [-1, hwc_shape])
        inputs = tf.reshape(y_pred, [-1, hwc_shape])
        
        weights=K.sum(targets,axis=1)/tf.cast(hwc_shape, dtype=tf.float32)
        weights=K.clip(1-weights,0.3,0.7)
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets),axis=1)
        FP = K.sum(((1-targets) * inputs),axis=1)
        FN = K.sum((targets * (1-inputs)),axis=1)
               
        Tversky = (TP + self.smooth) / (TP + weights*FP + weights*FN + self.smooth)  
        
        FocalTversky = K.mean(K.pow((1 - Tversky), 1/self.gamma))
        return FocalTversky

class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self,weights,from_logits=False,  **kwargs):
        super().__init__(**kwargs)
        self.weights=weights
        self.epsilon=K.epsilon()
        self.from_logits=from_logits
        
    def call(self,y_true, y_pred,) :
        if self.from_logits:
           y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        	#y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * self.weights
        loss = -K.sum(loss, -1)
        return K.mean(loss)

class GeneralizedDice(tf.keras.losses.Loss):
    def __init__(self, smooth=1,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=K.epsilon()


    def call(self, y_true, y_pred):

        # [b, h, w, classes]
        y_true_shape = tf.shape(y_true)

        # [b, h*w, classes]
        y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
        y_pred = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

        # [b, classes]
        # count how many of each class are present in 
        # each image, if there are zero, then assign
        # them a fixed weight of eps
        counts = tf.reduce_sum(y_true, axis=1)
        weights = 1. / (counts ** 2)
        weights = tf.where(tf.math.is_finite(weights), weights, self.epsilon)

        multed = tf.reduce_sum(y_true * y_pred, axis=1)
        summed = tf.reduce_sum(y_true + y_pred, axis=1)

        # [b]
        numerators = tf.reduce_sum(weights*multed, axis=-1)
        denom = tf.reduce_sum(weights*summed, axis=-1)
        dices = 1. - 2. * numerators / denom
        dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
        return tf.reduce_mean(dices)
    
class FocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(self,alpha=0.5,gamma=4/3, smooth=0.000001, **kwargs):
        super().__init__(**kwargs)
        self.smooth=smooth
        self.epsilon=K.epsilon()
        self.alpha = alpha
        self.gamma = gamma
    def call(self, y_true, y_pred):

        inputs = K.flatten(y_pred)
        targets = K.flatten(y_true)  
        #True Positives, False Positives & False Negatives
        TP = K.sum((inputs * targets))
        FP = K.sum(((1-targets) * inputs))
        FN = K.sum((targets * (1-inputs)))
               
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + (1-self.alpha)*FN + self.smooth)  
        FocalTversky = K.pow((1 - Tversky), 1/self.gamma)
        return FocalTversky


class AsymUnifiedFocalLoss(tf.keras.losses.Loss):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Ref : https://github.com/mlyg/unified-focal-loss
    Paper: Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation

    Deleted Background Class - enhancment etc. because we dont use
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5,**kwargs):
        super().__init__(**kwargs)
        self.epsilon=K.epsilon()
        self.weight=weight
        self.delta=delta
        self.gamma=gamma

    def asymmetric_focal_tversky_loss(self,y_true, y_pred,delta=0.7, gamma=0.75):
        # Clip values to prevent division by zero error
        epsilon = self.epsilon
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=[1,2])
        fn = K.sum(y_true * (1-y_pred), axis=[1,2])
        fp = K.sum((1-y_true) * y_pred, axis=[1,2])
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)


        fore_dice = (1-dice_class) * K.pow(1-dice_class, -gamma) 

        # Average class scores
        loss = K.mean(tf.stack([fore_dice],axis=-1))

        return loss
    
    def asymmetric_focal_loss(self,y_true, y_pred,delta=0.7, gamma=2.):


        epsilon = self.epsilon
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        fore_ce = delta * cross_entropy

        loss = K.mean(K.sum(tf.stack([fore_ce],axis=-1),axis=-1))

        return loss
    
    def call(self, y_true, y_pred):
        asymmetric_ftl = self.asymmetric_focal_tversky_loss(y_true,y_pred,delta=self.delta, gamma=self.gamma)
        asymmetric_fl = self.asymmetric_focal_loss(y_true,y_pred,delta=self.delta, gamma=self.gamma)
        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
        else:
            return asymmetric_ftl + asymmetric_fl


class GeneralisedWassersteinDiceLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def wasserstein_disagreement_map(self,prediction, ground_truth, M):
        """
        Function to calculate the pixel-wise Wasserstein distance between the
        flattened pred_proba and the flattened labels (ground_truth) with respect
        to the distance matrix on the label space M.
        :param prediction: the logits after softmax
        :param ground_truth: segmentation ground_truth
        :param M: distance matrix on the label space
        :return: the pixelwise distance map (wass_dis_map)
        """
        # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
        # wrt the distance matrix on the label space M
        n_classes = K.int_shape(prediction)[-1]
        # unstack_labels = tf.unstack(ground_truth, axis=-1)
        ground_truth = tf.cast(ground_truth, dtype=tf.float64)
        # unstack_pred = tf.unstack(prediction, axis=-1)
        prediction = tf.cast(prediction, dtype=tf.float64)
        # print("shape of M", M.shape, "unstacked labels", unstack_labels,
        #       "unstacked pred" ,unstack_pred)
        # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
        pairwise_correlations = []
        for i in range(n_classes):
            for j in range(n_classes):
                pairwise_correlations.append(
                    M[i, j] * tf.multiply(prediction[:,i], ground_truth[:,j]))
        wass_dis_map = tf.add_n(pairwise_correlations)
        return wass_dis_map


    def call(self,y_true, y_pred ):


        """
        Function to calculate the Generalised Wasserstein Dice Loss defined in
        Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
        Multi-class Segmentation using Holistic Convolutional Networks.
        MICCAI 2017 (BrainLes)
        :param prediction: the logits (before softmax)
        :param ground_truth: the segmentation ground_truth
        :param weight_map:
        :return: the loss
        """
        # apply softmax to pred scores
        n_classes = K.int_shape(y_pred)[-1]


        ground_truth = tf.cast(tf.reshape(y_true,(-1,n_classes)), dtype=tf.int64)
        pred_proba = tf.cast(tf.reshape(y_pred,(-1,n_classes)), dtype=tf.float64)

        # M = tf.cast(M, dtype=tf.float64)
        # compute disagreement map (delta)
        M = M_tree_4
        # print("M shape is ", M.shape, pred_proba, one_hot)
        delta = self.wasserstein_disagreement_map(pred_proba, ground_truth, M)
        # compute generalisation of all error for multi-class seg
        all_error = tf.reduce_sum(delta)
        # compute generalisation of true positives for multi-class seg
        one_hot = tf.cast(ground_truth, dtype=tf.float64)
        true_pos = tf.reduce_sum(
            tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
            axis=1)
        true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
        WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)

        return tf.cast(WGDL, dtype=tf.float32)