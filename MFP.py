from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,Concatenate,Input
def MFPCovBlock1(inputShape,filter,size,rate):
    x = Conv2D(filters=filter,kernel_size=size,dilation_rate=rate,padding="same")(inputShape)
    x= BatchNormalization()(x)
    x= Activation("relu")(x)
    return x

def MFPCovBlock2(inputShape,filter,size,size2,rate,rate2):
    x=MFPCovBlock1(inputShape,filter,size,rate)
    x=MFPCovBlock1(x,filter,size2,rate2)
    return x

def MFPCovBlockMain(inputShape,filter):
    s1=MFPCovBlock1(inputShape,filter,3,1)
    s2=MFPCovBlock1(inputShape,filter,3,3)
    s3=MFPCovBlock1(inputShape,filter,5,1)
    s4=MFPCovBlock2(inputShape,filter,3,1,1,1)
    s5=MFPCovBlock2(inputShape,filter,3,3,3,5)
    o=Concatenate()([s1,s2,s3,s4,s5])
    return o
if __name__ == "__main__":
    input_shape=(256,256,3)
    inp=Input(input_shape)
    mdl= MFPCovBlockMain(inp,32)