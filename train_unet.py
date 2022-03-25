import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from unet import UNet
from metrics_value import dice_loss, dice_coef,iou

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(path):
    x=sorted(glob(os.path.join(path,"images", "*.jpg")))
    y=sorted(glob(os.path.join(path,"groundtruth", "*.jpg")))
    return x,y

def shuffling(x,y): #numpy shuffling
    x,y=shuffle(x,y,random_state=42)
    return x,y

def read_image(path):
    path=path.decode()
    x=cv2.imread(path, cv2.IMREAD_COLOR)
    x=x/255.0
    x=x.astype(np.float32)
    return x

def read_groundtruth(path):
    path=path.decode()
    x=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x=x/255.0
    x=x.astype(np.float32)
    x=np.expand_dims(x,axis=-1)
    return x

def tf_parse(x,y):
    def parse(x,y):
        x=read_image(x)
        y=read_groundtruth(y)
        
        return x,y
    
    x,y=tf.numpy_function(parse,[x,y],[tf.float32,tf.float32])
    x.set_shape([256,256,3])
    y.set_shape([256,256,1])
    
    return x,y

def tf_dataset(x,y,batch_size=2):
    dataset= tf.data.Dataset.from_tensor_slices((x,y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    
    return dataset



if __name__ == "__main__":
    
    create_directory("file")
    
    batch=10
    lr= 1e-4
    epochs=100
    model_path=os.path.join("files","model.h5")
    path_csv=os.path.join("files","dataset.csv")
    
    dataset_path ="resized"
    train_path=os.path.join(dataset_path,"train") 
    valid_path=os.path.join(dataset_path,"test")
    
    xtrain,ytrain=load_data(train_path)
    xtrain,ytrain=shuffling(xtrain,ytrain) #shuffling the datas
    
    xvalid,yvalid=load_data(valid_path)
    xvalid,yvalid=shuffling(xvalid,yvalid)
    
    print(len(xvalid),len(yvalid),len(xtrain),len(ytrain))
    
    train_dataset = tf_dataset(xtrain,ytrain,batch_size=batch)
    valid_dataset = tf_dataset(xvalid,yvalid,batch_size=batch)
    
    train_steps=len(xtrain)//batch
    valid_steps=len(xvalid)//batch
    
    if len(xtrain) % batch !=0:
        train_steps +=1
    
    if len(xvalid) % batch !=0:
        valid_steps +=1  
        
    model = UNet((256,256,3))
    model.compile(loss=dice_loss,optimizer=Adam(lr),metrics=[dice_coef,iou,Recall(),Precision()])
    #model.summary()
    
    callbacks = [
        ModelCheckpoint(model_path,verbose=1,save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=5,min_lr=1e-6,verbose=1),
        CSVLogger(path_csv),
        TensorBoard(),
        EarlyStopping(monitor="val_loss",patience =10,restore_best_weights=False)
        ]
    
    model.fit(train_dataset,
              epochs=5,
              validation_data=valid_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              callbacks=callbacks)