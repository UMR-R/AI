import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input
from tensorflow.keras.models import Model

# 这里实现一个VGG网络，返回的是一个128维向量，用于siamese的输入
def VGG(X_input):
    X = X_input
    X = Conv2D(64, (3,3), padding = 'same')(X)
    X = Conv2D(64, (3,3), padding = 'same')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=2)(X)
    X = Conv2D(128, (3,3), padding = 'same')(X)
    X = Conv2D(128, (3,3), padding = 'same')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=2)(X)
    X = Conv2D(256, (3,3), padding = 'same')(X)
    X = Conv2D(256, (3,3), padding = 'same')(X)
    X = Conv2D(256, (3,3), padding = 'same')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=2)(X)
    X = Conv2D(512, (3,3), padding = 'same')(X)
    X = Conv2D(512, (3,3), padding = 'same')(X)
    X = Conv2D(512, (3,3), padding = 'same')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=2)(X)
    X = Flatten()(X)
    X = Dense(128)(X)
    
    X = Lambda(lambda  x: K.l2_normalize(x,axis=1))(X)
    return X

def VGG_Siamese(input_shape):
    X1_input = Input(input_shape)
    X2_input = Input(input_shape)

    X1 = ZeroPadding2D((3, 3))(X1_input)
    X2 = ZeroPadding2D((3, 3))(X2_input)
    X1 = VGG(X1)
    X2 = VGG(X2)

    l1_distance_layer = Lambda(
        lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_distance_layer([X1, X2])    

    X = Dense(512, activation='relu')(l1_distance)
    X = Dense(2, activation='softmax')(X)

    model = Model(inputs = [X1_input, X2_input], outputs = X)
    
    return model