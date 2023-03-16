import argparse    
from omegaconf import OmegaConf
import tensorflow as tf
from model.usegformer import USegFormer
from model.callbacks import *
import os
from data.dataloader import DataGen,EvalGen



parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("--config_path", type=str, required=True,help="Config Path")

args = vars(parser.parse_args())

conf = OmegaConf.load(args["config_path"])
tf.keras.utils.set_random_seed(conf.seed)

batch_size=conf.batch_size
epochs=conf.epochs
train_path=conf.train_path
test_path=conf.test_path
checkpoint_path=conf.checkpoint_path
img_size=conf.input_shape[1]
train_ds=DataGen(train_path,batch_size=batch_size,img_size=img_size,augmentation=True)
eval_Data=EvalGen(test_path)

model=USegFormer(conf,checkpoint_path=checkpoint_path,)

model.compile()
returned_epoch=model.load()

path_conf=os.path.join(checkpoint_path,"config.yaml")
with open(path_conf ,'w') as file:
       OmegaConf.save(config=conf, f=file)

callbacks=[
    LearningRateStepScheduler(conf.lr,step_warmup=conf.step_warmup),
    SaveCheckpoint(number_epoch=epochs, monitor="val_f1_total",per_epoch=None,initial_value_threshold=0.5,  mode="max",save_best=True),
    tf.keras.callbacks.TensorBoard(log_dir=checkpoint_path+"/logs",write_graph=False, profile_batch=5,histogram_freq=1,write_steps_per_second=True),
    tf.keras.callbacks.CSVLogger(os.path.join(checkpoint_path,"log.csv"), separator=",", append=True)

]
model.fit(train_ds,validation_data=eval_Data,epochs=epochs,initial_epoch=returned_epoch,callbacks=callbacks)


results=model.evaluate(eval_Data,return_dict=True,)
try:
    log_txt_path=os.path.join(checkpoint_path,"results.txt")
    with open(log_txt_path, 'w') as f:
        for key in list(results.keys()):
            string=key+" :  "+str(results[key])+"\n"
            f.write(string)
except Exception as error:
    print(error)
    pass    