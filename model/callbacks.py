
from tensorflow.python.platform import tf_logging as logging
from keras.utils import io_utils
import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.platform import tf_logging as logging
from keras.utils import io_utils

class SaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self,number_epoch, monitor="val_iou",per_epoch=None,initial_value_threshold=0.05,  mode: str = "min",save_best=False):
        super(SaveCheckpoint, self).__init__()
        
        self.per_epoch = per_epoch
        self.number_epoch=number_epoch
        self.monitor = monitor
        self.mode=mode
        self.best=initial_value_threshold
        self.save_best=save_best
        if mode not in ["min", "max"]:
            logging.warning(
                "ModelCheckpoint mode %s is unknown, fallback to min mode.",
                mode,
            )
            mode = "min"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf


    def save_best_model(self,epoch,logs):
        self.checkpoint_best_path= os.path.join(self.model.checkpoint_dir, "best")+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        current = logs.get(self.monitor)
        filepath=self.model.checkpoint_prefix
        if current is None:
            logging.warning(
                "Can save best model only with %s available, "
                "skipping.",
                self.monitor,
            )
        else:
            if self.monitor_op(current, self.best):
                io_utils.print_msg(
                    f"\nEpoch {epoch + 1}: {self.monitor} "
                    "improved "
                    f"from {self.best:.5f} to {current:.5f}, "
                    f"saving model to {filepath}"
                )
                self.best = current
                self.model.checkpoint.save(file_prefix = self.checkpoint_best_path)
 
    def on_epoch_end(self, epoch, logs=None):
        if self.save_best==True:
            self.save_best_model(epoch,logs)
        if self.per_epoch:
            if (epoch%self.per_epoch)==0 and epoch!=0:
                self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)
                io_utils.print_msg(
                    f"\nEpoch {epoch + 1} "
                    f"saving checkpoint to {self.model.checkpoint_prefix}"
                )
        if self.number_epoch-1==epoch:
            self.model.checkpoint.save(file_prefix = self.model.checkpoint_prefix)
            io_utils.print_msg(
                    f"\nFinal Epoch {epoch + 1} "
                    f"saving checkpoint to {self.model.checkpoint_prefix}")
                
        self.model.checkpoint.epoch.assign_add(1)



class LearningRateStepScheduler(tf.keras.callbacks.Callback):

    def __init__(self, initial_lr,step_warmup=7500, verbose=0):
        super(LearningRateStepScheduler, self).__init__()

        self.verbose = verbose
        self.step_warmup=step_warmup
        self.initial_lr=initial_lr
        self.steps_intervals=np.linspace(0, 40, 41,endpoint=True)*step_warmup
        
    def schedule(self,lr):
        if int(self.model.checkpoint.step)>self.step_warmup:
            rate=np.searchsorted(self.steps_intervals, int(self.model.checkpoint.step) ,side='right')
            lr=self.initial_lr*(tf.math.exp(-0.1)**(rate))
        return lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        lr = self.schedule(lr)
   
        if not isinstance(lr, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f"should be float. Got: {lr}"
            )
        if isinstance(lr, tf.Tensor) and not lr.dtype.is_floating:
            raise ValueError(
                f"The dtype of `lr` Tensor should be float. Got: {lr.dtype}"
            )
        keras.backend.set_value(self.model.optimizer.lr, keras.backend.get_value(lr))
        if self.verbose > 0:
            io_utils.print_msg(
                f"\nEpoch {epoch + 1}: LearningRateScheduler setting learning "
                f"rate to {lr}."
            )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = keras.backend.get_value(self.model.optimizer.lr)
        tf.summary.scalar('learning rate', data=logs["lr"], step=self.model.checkpoint.step)

