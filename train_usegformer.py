import argparse    
from omegaconf import OmegaConf
import tensorflow as tf
import os
from utils.utils import instantiate_from_config,make_dirs

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("--config_path", type=str, required=True,help="Config Path")

args = vars(parser.parse_args())

config = OmegaConf.load(args["config_path"])
tf.keras.utils.set_random_seed(config.seed)

make_dirs(os.path.join(config.model.checkpoint_path,"checkpoint"))

checkpoint_path=os.path.join(config.model.checkpoint_path,"checkpoint")

train_ds=instantiate_from_config(config.data.train)
test_ds=instantiate_from_config(config.data.test)

model = instantiate_from_config(config.model)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate),
)
#history = model.fit(train_ds, validation_data=eval_Data, epochs=35,callbacks=callbacks)

def scheduler(epoch, lr):
    if epoch==0:
      return 0.0001
    elif epoch % 5==0 and epoch!=0 :
      return lr* (tf.math.exp(-0.1)**4)
    else:
      return lr 

callbacks=[]
callbacks.append( tf.keras.callbacks.LearningRateScheduler(scheduler))

callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(
      os.path.join(config.model.checkpoint_path,"usegformer"),
      monitor = "val_iou_total",
      save_best_only=True,
      save_weights_only = True,
      initial_value_threshold=0.35,
      mode='max'
  )
)

callbacks.append(
        tf.keras.callbacks.CSVLogger(os.path.join(config.model.checkpoint_path,"log.csv"), separator=",", append=True)

)
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config.model.checkpoint_path,"logs"),write_graph=False, profile_batch=5,histogram_freq=1,write_steps_per_second=True))
model.fit(train_ds, validation_data=test_ds, epochs=config.model.epochs,callbacks=callbacks)