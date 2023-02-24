
import albumentations as A
import tensorflow as tf
import numpy as np
import cv2
from data.utils import *


class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, path_list,
                 batch_size,img_size=512,resize_or_crop="resize"
                 ):
      
        pre_dis_files,post_dis_files,pre_target_files,post_target_files=get_idx_all_path(path_list)

        self.pre_dis_files=pre_dis_files
        self.pre_target_files=pre_target_files
        self.post_dis_files=post_dis_files
        self.post_target_files=post_target_files
        self.n = len(self.post_dis_files)
        self.batch_size=batch_size
        self.img_size=img_size
        self.transform = A.Compose([
              A.RandomSizedCrop(min_max_height=(img_size,img_size),width=img_size, height=img_size,always_apply=True),
              A.HorizontalFlip(p=0.5),
              A.RandomBrightnessContrast(p=0.2), ],
              additional_targets={"image1": "image","mask1": "mask"},


         )
 
    def __load_data__(self,i):  
        pre_dis=cv2.imread(self.pre_dis_files[i],cv2.IMREAD_COLOR)
        post_dis=cv2.imread(self.post_dis_files[i],cv2.IMREAD_COLOR)
        post_target=create_inference_image(self.post_target_files[i])
        pre_target=create_inference_image(self.pre_target_files[i])

        #post_target[post_target==5]=1

        transformed = self.transform(image=pre_dis,image1=post_dis, mask=post_target,mask1=pre_target)

        pre_dis_aug = transformed['image']
        post_dis_aug = transformed['image1']
        post_target_aug = transformed['mask']
        pre_target_aug = transformed['mask1']

        post_target_aug=get_encoded(post_target_aug)
        post_target_aug=np.float32(post_target_aug)
        pre_target_aug=np.expand_dims(np.float32(pre_target_aug),axis=-1)

        post_dis_aug=np.float32(post_dis_aug/255)
        pre_dis_aug=np.float32(pre_dis_aug/255)
        pre_target_aug[np.isnan(pre_target_aug)] = 0
        post_target_aug[np.isnan(post_target_aug)] = 0

        return pre_dis_aug,post_dis_aug,pre_target_aug,post_target_aug



    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_pres=[]
        batch_posts=[]
        batch_post_targets=[]
        batch_pre_targets=[]

        for i in range(i_start,i_end):
          pre_dis,post_dis,pre_target,post_target=self.__load_data__(i)
          batch_pres.append(pre_dis)
          batch_posts.append(post_dis)
          batch_post_targets.append(post_target)
          batch_pre_targets.append(pre_target)

        return np.asarray(batch_pres),np.asarray(batch_posts),np.asarray(batch_pre_targets),np.asarray(batch_post_targets)

    def __getitem__(self, index):
        pres,posts,pre_targets,post_targets=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return (pres,posts),(pre_targets,post_targets)
    
    def __len__(self):
        return self.n // self.batch_size