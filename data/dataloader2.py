
import albumentations as A
import tensorflow as tf
import numpy as np
import cv2
from data.utils import *
import random
import os
from skimage.morphology import square, dilation

class DataGen(tf.keras.utils.Sequence):
    
    def __init__(self, path_list,
                 batch_size,img_size=512,augmentation=False,
                 ):
      
        pre_dis_files,post_dis_files,post_target_files,pre_target_files=get_idx_all_path(path_list)

        self.pre_dis_files=pre_dis_files
        self.pre_target_files=pre_target_files
        self.post_dis_files=post_dis_files
        self.post_target_files=post_target_files
        self.n = len(self.post_dis_files)
        self.batch_size=batch_size
        self.img_size=img_size
        self.augmentation=augmentation
        self.target_source_path=os.path.join(path_list[0],"targets")

        self.transform = A.Compose([
              A.CropNonEmptyMaskIfExists (width=img_size, height=img_size,always_apply=True),
              A.RandomRotate90(p=0.2),
              A.Flip(p=0.6),

              A.OneOf([
                  A.MotionBlur(p=0.2),
                  A.MedianBlur(blur_limit=3, p=0.1),
                  A.Blur(blur_limit=3, p=0.1),
              ], p=0.2),
              A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
              A.OneOf([
                  A.OpticalDistortion(p=0.3),
                  A.GridDistortion(p=0.1),
              ], p=0.2),        
              A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.2),
              A.RandomBrightnessContrast(p=0.2), ],
              additional_targets={"image1": "image","mask1": "mask"},

         )
        self.transform_no_aug= A.Compose([
              A.RandomSizedCrop(min_max_height=(img_size,img_size),width=img_size, height=img_size,always_apply=True)],
              additional_targets={"image1": "image","mask1": "mask"},
        )
 
    def __load_data__(self,i):  
        pre_dis=cv2.imread(self.pre_dis_files[i],cv2.IMREAD_COLOR)
        post_dis=cv2.imread(self.post_dis_files[i],cv2.IMREAD_COLOR)
        target_path=self.post_dis_files[i].split("/")
        target_path=target_path[-1]
        target_path=target_path.replace("post_disaster","post_disaster_target")
        post_target=cv2.imread(os.path.join(self.target_source_path,target_path),cv2.IMREAD_UNCHANGED)
        pre_target=cv2.imread(os.path.join(self.target_source_path,target_path.replace(("post_disaster","pre_disaster"))),cv2.IMREAD_UNCHANGED)
        msk0=pre_target
        msk1 = np.zeros_like(post_target)
        msk2 = np.zeros_like(post_target)
        msk3 = np.zeros_like(post_target)
        msk4 = np.zeros_like(post_target)
        msk1[post_target == 1] = 255
        msk2[post_target == 2] = 255
        msk3[post_target == 3] = 255
        msk4[post_target == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        msk[..., 0] = True
        msk[..., 1][msk[..., 2:].max(axis=2)] = False
        msk[..., 3][msk[..., 2]] = False
        msk[..., 4][msk[..., 2]] = False
        msk[..., 4][msk[..., 3]] = False
        msk[..., 0][msk[..., 1:].max(axis=2)] = False
        msk = msk * 1

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


        post_dis_aug=np.float32((post_dis_aug/127)-1)
        pre_dis_aug=np.float32((pre_dis_aug/127)-1)

        post_target_onehot_aug[np.isnan(post_target_onehot_aug)] = 0
        post_target_label_aug[np.isnan(post_target_label_aug)] = 0

        return pre_dis_aug,post_dis_aug,post_target_onehot_aug,post_target_label_aug



    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_pres=[]
        batch_posts=[]
        batch_post_label_targets=[]
        batch_post_onehot_targets=[]

        for i in range(i_start,i_end):
          pre_dis,post_dis,post_target_onehot,post_target_onehot,post_target_label=self.__load_data__(i)
          batch_pres.append(pre_dis)
          batch_posts.append(post_dis)
          batch_post_onehot_targets.append(post_target_onehot)
          batch_post_label_targets.append(post_target_label)

        return np.asarray(batch_pres),np.asarray(batch_posts),np.asarray(batch_post_onehot_targets),np.asarray(batch_post_label_targets)

    def __getitem__(self, index):
        pres,posts,post_target_onehots,post_target_labels=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return (pres,posts),(post_target_onehots,post_target_labels)
    
    def __len__(self):
        return self.n // self.batch_size
    




class EvalGen(tf.keras.utils.Sequence):
    
    def __init__(self, path_list,img_size=512,
                 ):
      
        pre_dis_files,post_dis_files,post_target_files,pre_target_files=get_idx_all_path(path_list)

        self.pre_dis_files=pre_dis_files
        self.pre_target_files=pre_target_files
        self.post_dis_files=post_dis_files
        self.post_target_files=post_target_files
        self.n = len(self.post_dis_files)
        self.batch_size=1
        self.img_size=img_size
        self.target_source_path=os.path.join(path_list[0],"targets")

    def split2piece(self,im):
        #To Do 
        #split other for imgsize
        if len(im.shape)==3:
            pieces=[im[:512,:512,:],im[:512,512:,:],im[512:,:512,:],im[512:,512:,:]]

        elif len(im.shape)==2:
            pieces=[im[:512,:512],im[:512,512:],im[512:,:512],im[512:,512:]]
        else:
            raise("Invalid Shape On Masks")

        return pieces
    
    def __load_data__(self,i):  
        pre_dis=cv2.imread(self.pre_dis_files[i],cv2.IMREAD_COLOR)
        post_dis=cv2.imread(self.post_dis_files[i],cv2.IMREAD_COLOR)
        target_path=self.post_dis_files[i].split("/")
        target_path=target_path[-1]
        target_path=target_path.replace("post_disaster","post_disaster_target")
        post_target=cv2.imread(os.path.join(self.target_source_path,target_path),cv2.IMREAD_UNCHANGED)
        pre_target=create_inference_image(self.pre_target_files[i])
        msk0=pre_target
        msk1 = np.zeros_like(post_target)
        msk2 = np.zeros_like(post_target)
        msk3 = np.zeros_like(post_target)
        msk4 = np.zeros_like(post_target)
        msk1[post_target == 1] = 255
        msk2[post_target == 2] = 255
        msk3[post_target == 3] = 255
        msk4[post_target == 4] = 255

        msk0 = msk0[..., np.newaxis]
        msk1 = msk1[..., np.newaxis]
        msk2 = msk2[..., np.newaxis]
        msk3 = msk3[..., np.newaxis]
        msk4 = msk4[..., np.newaxis]

        msk = np.concatenate([msk0, msk1, msk2, msk3, msk4], axis=2)
        msk = (msk > 127)

        msk = msk * 1

        post_target = msk[..., 1:].argmax(axis=2)

        pre_dis=np.float32((pre_dis/127)-1)
        post_dis=np.float32((post_dis/127)-1)
        return self.split2piece(pre_dis),self.split2piece(post_dis),self.split2piece(msk),self.split2piece(post_target)


    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_pres=[]
        batch_posts=[]
        batch_post_label_targets=[]
        batch_post_onehot_targets=[]

        for i in range(i_start,i_end):
          pre_dis,post_dis,post_target_onehot,post_target_onehot,post_target_label=self.__load_data__(i)
          batch_pres.append(pre_dis)
          batch_posts.append(post_dis)
          batch_post_onehot_targets.append(post_target_onehot)
          batch_post_label_targets.append(post_target_label)

        return np.asarray(batch_pres),np.asarray(batch_posts),np.asarray(batch_post_onehot_targets),np.asarray(batch_post_label_targets)

    def __getitem__(self, index):
        pres,posts,post_target_onehots,post_target_labels=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return (pres,posts),(post_target_onehots,post_target_labels)
    
    def __len__(self):
        return self.n // self.batch_size
    
class UnetDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, path_list,
                 batch_size,img_size=512,dilation=True,augmentation=False,
                 ):
      
        pre_dis_files,post_dis_files,post_target_files,pre_target_files=get_idx_all_path(path_list)

        self.mask_files=pre_target_files
        self.image_files=pre_dis_files
        self.dilation=dilation
        self.augmentation=augmentation
        self.n = len(self.image_files)
        self.batch_size=batch_size
        self.img_size=img_size
        self.transform = A.Compose([
              A.CropNonEmptyMaskIfExists (width=img_size, height=img_size,always_apply=True),
              A.RandomRotate90(p=0.6),
              A.Flip(p=0.6),

              A.OneOf([
                  A.MotionBlur(p=0.2),
                  A.MedianBlur(blur_limit=3, p=0.1),
                  A.Blur(blur_limit=3, p=0.1),
              ], p=0.2),
              A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
              A.OneOf([
                  A.OpticalDistortion(p=0.3),
                  A.GridDistortion(p=0.1),
              ], p=0.2),        
              A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.2),
              A.RandomBrightnessContrast(p=0.2), ],

         )
        self.transform_no_aug= A.Compose([
              A.RandomSizedCrop(min_max_height=(img_size,img_size),width=img_size, height=img_size,always_apply=True),])
 
    def __load_data__(self,i):  
        image=cv2.imread(self.image_files[i],cv2.IMREAD_COLOR)
        mask=create_inference_image(self.mask_files[i])
        mask=(mask>0.25)*1
        if self.dilation==True:
          mask=np.uint8(mask*255)
          kernel = np.ones((5,5),np.uint8)
          mask = cv2.dilate( mask ,kernel,iterations = 1)
          mask=(mask>127)*1

        if self.augmentation==True:
            if random.random() > 0.7:
                transformed = self.transform(image=image,mask=mask)
            else:
                transformed = self.transform_no_aug(image=image,mask=mask)
        else:
            transformed = self.transform_no_aug(image=image,mask=mask)

        image = transformed['image']
        mask = transformed['mask']


        mask=np.expand_dims(np.float32(mask),axis=-1)

        image=np.float32(image/255)

        mask[np.isnan(mask)] = 0

        return np.asarray(image),np.asarray(mask)


    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_img=[]
        batch_mask=[]


        for i in range(i_start,i_end):
          image,mask=self.__load_data__(i)
          batch_mask.append(mask)
          batch_img.append(image)

        return np.asarray(batch_img),np.asarray(batch_mask)
    

    def __getitem__(self, index):
        images,masks=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return (np.asarray(images)),(np.asarray(masks))
    
    def __len__(self):
        return self.n // self.batch_size
    
class EvalUnetGen(tf.keras.utils.Sequence):
    
    def __init__(self, path_list,img_size=512,
                 ):
      
        pre_dis_files,post_dis_files,post_target_files,pre_target_files=get_idx_all_path(path_list)

        self.pre_dis_files=pre_dis_files
        self.pre_target_files=pre_target_files
        self.post_dis_files=post_dis_files
        self.post_target_files=post_target_files
        self.n = len(self.post_dis_files)
        self.batch_size=1
        self.img_size=img_size

    def split2piece(self,im):
        #To Do 
        #split other for imgsize
        if len(im.shape)==3:
            pieces=[im[:512,:512,:],im[:512,512:,:],im[512:,:512,:],im[512:,512:,:]]

        elif len(im.shape)==2:
            pieces=[im[:512,:512],im[:512,512:],im[512:,:512],im[512:,512:]]
        else:
            raise("Invalid Shape On Masks")

        return pieces
    
    def __load_data__(self,i):  
        pre_dis=cv2.imread(self.pre_dis_files[i],cv2.IMREAD_COLOR)
        #post_dis=cv2.imread(self.post_dis_files[i],cv2.IMREAD_COLOR)
        #post_target=create_inference_image(self.post_target_files[i])
        pre_target=create_inference_image(self.pre_target_files[i])
        pre_target=(pre_target>0.25)*1

        #post_target=get_encoded(post_target)
        #post_target=np.float32(post_target)
        pre_target=np.expand_dims(np.float32(pre_target),axis=-1)

        #post_dis=np.float32(post_dis/255)
        pre_dis=np.float32(pre_dis/255)
        pre_target[np.isnan(pre_target)] = 0
        #post_target[np.isnan(post_target)] = 0

        return self.split2piece(pre_dis),self.split2piece(pre_target)



    def __get_batch__(self,index_interval):
        i_start=index_interval[0]
        i_end=index_interval[1]
        batch_pres=[]
        batch_pre_targets=[]

        for i in range(i_start,i_end):
          pre_dis,pre_target=self.__load_data__(i)
          batch_pres.extend(pre_dis)

          batch_pre_targets.extend(pre_target)

        return np.asarray(batch_pres),np.asarray(batch_pre_targets)

    def __getitem__(self, index):
        pres,pre_targets=self.__get_batch__(index_interval=[index * self.batch_size,(index + 1) * self.batch_size])
        return (pres),(pre_targets)
    
    def __len__(self):
        return self.n // self.batch_size
    
