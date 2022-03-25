#..........import required modules................
#import tensorflow as tf
#from keras.layers import Convolution2D as Conv2D
#from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D,Activation ,MaxPool2D, UpSampling2D,Concatenate,BatchNormalization,Conv2DTranspose
from MFP import MFPCovBlockMain
#from keras.optimizers import adam_v2
#import os
#from keras.layers import Activation, MaxPool2D, Concatenate

#...........to be passed to the respective decoder layer via skip connections............

def conv_block(inputs, num_filters):
    x= Conv2D(num_filters,3,padding='same')(inputs)
    x=BatchNormalization()(x)
    x= Activation("relu")(x)

    x= Conv2D(num_filters,3,padding='same')(x)
    x=BatchNormalization()(x)
    x= Activation("relu")(x)

    return x

#............to be passed to next encoder layer...............

def encoder(inputs, num_filters):
    x= conv_block(inputs,num_filters)
    p= MaxPool2D((2,2))(x)
    return x,p

#.............to be passed to next decoder layer.............

def decoder(inputs,skip_connections,num_filters):
    x= Conv2DTranspose(num_filters,2, strides=2,padding="same")(inputs)
    x= Concatenate()([x,skip_connections])
    x= conv_block(x,num_filters)

    return x

#.............Arcitecture of the Unet modle.............

# def UNet(input_shape):
#     inputs= Input(input_shape)
#     s1,p1= encoder(inputs,64)
#     s2,p2= encoder(p1,128)
#     s3,p3= encoder(p2,256)
#     s4,p4= encoder(p3,512)

#     b1 = conv_block(p4,1024)

#     d1= decoder(b1,s4,512)
#     d2= decoder(d1,s3,256)
#     d3= decoder(d2,s2,128)
#     d4= decoder(d3,s1,64)

#     outputs = Conv2D(1,1,padding='same',activation='sigmoid')(d4)

#     model=Model(inputs,outputs,name='UNET')
#     return model  
#  

if __name__ == "__main__":
    input_shape=(256,256,3)
    mdl= UNet(input_shape)
    mdl.summary()