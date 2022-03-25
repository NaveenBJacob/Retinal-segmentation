#data preparation and data augmentation

import os
#import numpy as np
import cv2
from glob import glob
import imageio


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def load_data():
    
    #x is for the images and y is for the ground truth returning the list image names
    
    xtrain=sorted(glob(os.path.join("A. Segmentation","1. Original Images","a. Training Set", "*.jpg")))
    ytrain=sorted(glob(os.path.join("A. Segmentation","2. All Segmentation Groundtruths","a. Training Set","5. Optic Disc", "*.tif")))
    
    xtest=sorted(glob(os.path.join("A. Segmentation","1. Original Images","b. Testing Set", "*.jpg")))
    ytest=sorted(glob(os.path.join("A. Segmentation","2. All Segmentation Groundtruths","b. Testing Set","5. Optic Disc", "*.tif")))
    return ((xtrain,ytrain),(xtest,ytest))

#.........Function to make image size to 256X256 (for OD segmentation)...........
def resize(image,ground_truth,path,flag=True):
    
    h,w=256,256
    #iteraation through the entire training set(imgs+ground truths) as img size==ground truths
    for i in range(len(image)): 
        x= image[i]
        y=ground_truth[i]
        name=list(os.path.split(x))[-1].split('.')[0]
        
        img=cv2.imread(x,cv2.IMREAD_COLOR)
        grnd=cv2.imread(y,cv2.IMREAD_GRAYSCALE)

        if flag == True:
            pass
        else:
    
            X=[img]
            Y=[grnd]
        
        val=0
        for a,b in zip(X,Y):
            
            a=cv2.resize(a,(w,h))
            b=cv2.resize(b,(w,h))

            
            if len(Y)==1:
                img_name=f"n{name}.jpg"
                grnd_name=f"n{name}.jpg"
            else:
                img_name=f"n{name}_{val}.jpg"
                grnd_name=f"n{name}_{val}.jpg"
            
            path_image=os.path.join(path,"images",img_name)
            path_grndtruth=os.path.join(path,"groundtruth",grnd_name)
            
            cv2.imwrite(path_image,a)
            cv2.imwrite(path_grndtruth,b)
            val+=1
        
        

if __name__ == "__main__":
    
    (xtrain,ytrain),(xtest,ytest) = load_data()
    print(len(xtrain),len(ytrain),len(xtest),len(ytest))
    
    create_directory("resized/train/images")
    create_directory("resized/train/groundtruth")
    create_directory("resized/test/images")
    create_directory("resized/test/groundtruth")
    path="resized/train/"
    resize(xtrain,ytrain,path,flag=False)
    path="resized/test/"
    resize(xtest,ytest,path,flag=False)
    
    
    
    