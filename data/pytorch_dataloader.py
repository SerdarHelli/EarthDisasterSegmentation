
import albumentations as A
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from data.utils import *
import random
import os



def get_encodedx(img):
  mask = np.zeros(( *img.shape[:2],5))
  for i in range(0, 5):
    mask[ img[:, :] == i,i ] = 1

  return np.float32(mask)

class DataGen(Dataset):
    
    def __init__(self, path_list,
                 img_size=512,augmentation=False,
                 ):
      
        pre_dis_files,post_dis_files,post_target_files,_=get_idx_all_path(path_list)
        self.post_target_files=post_target_files

        self.pre_dis_files=pre_dis_files
        self.post_dis_files=post_dis_files
        self.n = len(self.post_dis_files)
        self.img_size=img_size
        self.augmentation=augmentation
        self.target_source_path=os.path.join(path_list[0],"targets")

        self.transform = A.Compose([
              A.CropNonEmptyMaskIfExists (width=img_size, height=img_size,always_apply=True),
              A.RandomRotate90(p=0.2),
              A.Flip(p=0.4),

              A.OneOf([
                  A.MotionBlur(p=0.2),
                  A.MedianBlur(blur_limit=1, p=0.1),
                  A.Blur(blur_limit=1, p=0.1),
              ], p=0.2),
              A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
              A.OneOf([
                  A.OpticalDistortion(p=0.3),
                  A.GridDistortion(p=0.1),
              ], p=0.2),        
              A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.2),
              A.RandomBrightnessContrast(p=0.2), ],
              additional_targets={"image1": "image","mask1": "mask"},

         )
        self.transform_no_aug= A.Compose([
              A.Resize(width=img_size, height=img_size,always_apply=True)],
              additional_targets={"image1": "image","mask1": "mask"},
        )
 
    def __getitem__(self,i):  
        pre_dis=cv2.imread(self.pre_dis_files[i],cv2.IMREAD_COLOR)
        post_dis=cv2.imread(self.post_dis_files[i],cv2.IMREAD_COLOR)
        target_path=self.post_dis_files[i].split("/")
        target_path=target_path[-1]
        target_path=target_path.replace("post_disaster","post_disaster_target")
        post_target= create_inference_image(self.post_target_files[i])

        #pre_target=cv2.imread(os.path.join(self.target_source_path,target_path.replace("post_disaster","pre_disaster")),cv2.IMREAD_UNCHANGED)
        msk=get_encodedx(post_target)
        post_target_label = msk.argmax(axis=2)
        post_target_onehot = msk


        if self.augmentation==True:
            transformed = self.transform(image=pre_dis,image1=post_dis, mask=post_target_onehot,mask1=post_target_label)
            if random.random() > 0.7:
                transformed = self.transform(image=pre_dis,image1=post_dis, mask=post_target_onehot,mask1=post_target_label)
            else:
                transformed = self.transform_no_aug(image=pre_dis,image1=post_dis, mask=post_target_onehot,mask1=post_target_label)

        else:
            transformed = self.transform_no_aug(image=pre_dis,image1=post_dis, mask=post_target_onehot,mask1=post_target_label)

        pre_dis_aug = transformed['image']
        post_dis_aug = transformed['image1']
        post_target_onehot_aug = transformed['mask']
        post_target_label_aug = transformed['mask1']

        post_target_onehot_aug=np.float32(post_target_onehot_aug)
        post_target_label_aug=np.float32(post_target_label_aug)


        post_dis_aug=np.float32(post_dis_aug/255)
        pre_dis_aug=np.float32(pre_dis_aug/255)

        post_target_onehot_aug[np.isnan(post_target_onehot_aug)] = 0
        post_target_label_aug[np.isnan(post_target_label_aug)] = 0


        pre_dis_aug=np.transpose(pre_dis_aug, (2,0,1))
        post_dis_aug=np.transpose(post_dis_aug, (2,0,1))
        post_target_onehot_aug=np.transpose(post_target_onehot_aug, (2,0,1))

        return pre_dis_aug,post_dis_aug,post_target_onehot_aug,post_target_label_aug


    def __len__(self):
        return self.n 
    


class EvalGen(Dataset):
    
    def __init__(self, path_list,img_size=512,
                 ):
      
        pre_dis_files,post_dis_files,post_target_files,pre_target_files=get_idx_all_path(path_list)
        self.post_target_files=post_target_files
        self.pre_target_files=pre_target_files

        self.pre_dis_files=pre_dis_files
        self.post_dis_files=post_dis_files
        self.n = len(self.post_dis_files)
        self.img_size=img_size
        self.target_source_path=os.path.join(path_list[0],"targets")
        self.transform_no_aug= A.Compose([
              A.Resize(width=img_size, height=img_size,always_apply=True)],
              additional_targets={"image1": "image","mask1":"mask"},
        )
    def split2piece(self,im):
        #To Do 
        #split other for imgsize
        if len(im.shape)==3:
            pieces=[im[:512,:512,:],im[:512,512:,:],im[512:,:512,:],im[512:,512:,:]]

        elif len(im.shape)==2:
            pieces=[im[:512,:512],im[:512,512:],im[512:,:512],im[512:,512:]]
        else:
            raise("Invalid Shape On Masks")
        return np.asarray(pieces)


    def resize(self,im):
        new_list=[]
        for i in range(len(im)):
            c=cv2.resize(im[i],(self.img_size,self.img_size),interpolation = cv2.INTER_AREA)
            new_list.append(c)

        return np.asarray(new_list)


    def __getitem__(self,i):  
        pre_dis=cv2.imread(self.pre_dis_files[i],cv2.IMREAD_COLOR)
        post_dis=cv2.imread(self.post_dis_files[i],cv2.IMREAD_COLOR)
        target_path=self.post_dis_files[i].split("/")
        target_path=target_path[-1]
        target_path=target_path.replace("post_disaster","post_disaster_target")
        post_target= create_inference_image(self.post_target_files[i])
        pre_target= create_inference_image(self.pre_target_files[i])

        #pre_target=cv2.imread(os.path.join(self.target_source_path,target_path.replace("post_disaster","pre_disaster")),cv2.IMREAD_UNCHANGED)
        msk=get_encodedx(post_target)
        post_target = msk.argmax(axis=2)


        pre_dis=self.split2piece(pre_dis)
        post_dis=self.split2piece(post_dis)
        post_target=self.split2piece(post_target)
        pre_target=self.split2piece(pre_target)

        post_dis=np.float32(post_dis/255)
        pre_dis=np.float32(pre_dis/255)
        pre_dis=np.transpose(pre_dis,(0,3,1,2))
        post_dis=np.transpose(post_dis,(0,3,1,2))

        return pre_dis,post_dis,pre_target,post_target
    
    def __len__(self):
        return self.n 


