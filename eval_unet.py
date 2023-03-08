import argparse    
from omegaconf import OmegaConf
import tensorflow as tf
from model.unet import UNetModel
from data.dataloader import EvalUnetGen
from sklearn.metrics import f1_score,jaccard_score
import numpy as np

parser = argparse.ArgumentParser(prog="Train")
parser.add_argument("--config_path", type=str, required=True,help="Config Path")

args = vars(parser.parse_args())

conf = OmegaConf.load(args["config_path"])

batch_size=conf.batch_size
epochs=conf.epochs
train_path=conf.train_path
test_path=conf.test_path
checkpoint_path=conf.checkpoint_path
img_size=conf.input_shape[1]

eval_Data=EvalUnetGen(test_path)

model=UNetModel(conf,checkpoint_path=checkpoint_path)


f1_scores=[]
jaccard_scores=[]
for i in range(eval_Data.__len__()):
    (x),(y_true)=eval_Data.__getitem__(i)
    y_pred=model.network.predict(x)
    f1_scores.append(f1_score(y_true.flatten(), y_pred.flatten(), average=None))
    jaccard_scores.append(jaccard_score(y_true.flatten(), y_pred.flatten(), average=None))


print("F1 Score : " , np.mean(np.asarray(f1_scores)))

print("Jaccard Score : " , np.mean(np.asarray(jaccard_scores)))
