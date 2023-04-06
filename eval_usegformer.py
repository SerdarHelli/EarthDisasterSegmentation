import argparse    
from omegaconf import OmegaConf
import tensorflow as tf
import os
from utils.utils import instantiate_from_config,make_dirs
import numpy as np

import pandas as pd
from tqdm import tqdm

def compute_tp_fn_fp(y_true, y_pred, c=1) :
    """
    Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)
    Args:
        y_pred (np.ndarray): prediction
        y_true (np.ndarray): target
        c (int): positive class
    """
    targ=y_true
    pred=y_pred



    TP = np.logical_and(pred == c, targ == c).sum()
    FN = np.logical_and(pred != c, targ == c).sum()
    FP = np.logical_and(pred == c, targ != c).sum()
    
    R=np.float32((TP+1e-6)/(TP+FN+1e-6))
    P=np.float32((TP+1e-6)/(TP+FP+1e-6))

    return np.float32((2*P*R)/(P+R))

harmonic_mean = lambda xs: len(xs) / sum((x+1e-6)**-1 for x in xs)

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("--config_path", type=str, required=True,help="Config Path")

args = vars(parser.parse_args())

config = OmegaConf.load(args["config_path"])
tf.keras.utils.set_random_seed(config.seed)

make_dirs(os.path.join(config.model.checkpoint_path,"checkpoint"))

checkpoint_path=os.path.join(config.model.checkpoint_path,"checkpoint")

test_ds=instantiate_from_config(config.data.test)
loc_eval_Data=instantiate_from_config(config.data.local)

model = instantiate_from_config(config.model)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.model.learning_rate),
)
#history = model.fit(train_ds, validation_data=eval_Data, epochs=35,callbacks=callbacks)
model.load_weights(os.path.join(config.model.checkpoint_path,"usegformer"))


iterator = tqdm(range(0,loc_eval_Data.__len__()), desc='Local Evualation', total=loc_eval_Data.__len__())

data_loc_results=pd.DataFrame(columns=["F1Total"])
for i, step in enumerate(iterator):
    (x,y),(x3)=loc_eval_Data.__getitem__(i)
    xt=np.transpose(x, (0, 3, 1,2))
    yt=np.transpose(y, (0, 3, 1,2))
    _,k=model.predict([xt,yt],verbose=0)
    pred=(k.argmax(axis=-1)>0.25)*1
    ground_true=x3
    for j  in range(pred.shape[0]):
        f1_total = compute_tp_fn_fp( ground_true[j,:,:,0],pred[j,:,:], 1)
        data={
            "F1Total":f1_total,

        }
        data_loc_results=data_loc_results.append(data,ignore_index=True)


data_loc_results.to_csv(os.path.join(config.model.checkpoint_path,"/loc_result.csv"), encoding='utf-8')





iterator = tqdm(range(0,test_ds.__len__()), desc='Multi Class Evualation', total=test_ds.__len__())

data_results=pd.DataFrame(columns=["F1Total","F1NoDamage","F1MinorDamage","F1MajorDamage","F1Destroyed"])
for i, step in enumerate(iterator):
    (x,y),(x2,x3)=test_ds.__getitem__(i)
    xt=np.transpose(x, (0, 3, 1,2))
    yt=np.transpose(y, (0, 3, 1,2))
    _,k=model.predict([xt,yt],verbose=0)
    pred=k.argmax(axis=-1)
    ground_true=x2.argmax(axis=-1)
    for j  in range(pred.shape[0]):
        f1_one_step = []
        for i in range(1,5): f1_one_step.append(compute_tp_fn_fp( ground_true[j,:,:],pred[j,:,:], i))
        f1_total = harmonic_mean(f1_one_step)
        data={
            "F1Total":f1_total,
            "F1NoDamage":f1_one_step[0],
            "F1MinorDamage":f1_one_step[1],
            "F1MajorDamage":f1_one_step[2],
            "F1Destroyed":f1_one_step[3]
        }
        data_results=data_results.append(data,ignore_index=True)


data_results.to_csv(os.path.join(config.model.checkpoint_path,"/multiclass_result.csv"), encoding='utf-8')
