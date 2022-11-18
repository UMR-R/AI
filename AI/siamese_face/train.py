from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import sys
from keras.preprocessing import image
from keras.utils import to_categorical
from load_utils import load_dataset

K.set_image_data_format('channels_first')

import time
import random
import cv2
import os
import warnings
import numpy as np
import tensorflow as tf
from vgg_net import *
from inception_blocks import *

warnings.filterwarnings("ignore")

np.set_printoptions(threshold=sys.maxsize)

x_train, y_train = load_dataset() # 这里的x维度是(m,nh,nw,nc),y的维度是(1,m)
# 为了适应模型输入，转换为通道优先
# x_train, y_train = np.moveaxis(x_train,-1,1), np.moveaxis(y_train,-1,0)
y_train = np.moveaxis(y_train,-1,0)
print(x_train[0].shape)

# 将数据集转换为一对一对的数据
lhs = []
rhs = []
labels = []
for i in range(x_train.shape[0]):
    for j in range(4):  # 每个数据与四个其他数据对比
        a = random.randint(0,x_train.shape[0]-1)
        while a == i:
            a = random.randint(0,x_train.shape[0]-1)
        lhs.append(x_train[i])
        rhs.append(x_train[a])
        if y_train[i] == y_train[a]:
            labels.append(1)
        else:
            labels.append(0)
            
# 因为后面将使用categorical_crossentropy作为损失函数，因此标签应当是
# 二维：(result, classes), 也是对应了softmax激活函数
labels = to_categorical(np.array(labels), num_classes=2)
# tensorflow中要接受带有shape方法的数据
lhs = np.array(lhs)
rhs = np.array(rhs)

# 获取模型
print("获取模型中...")
FRmodel = VGG_Siamese(input_shape=x_train[0].shape)
print("获取模型完毕！参数总量为："+str(FRmodel.count_params()))

# 设置模型参数
FRmodel.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
epoch = 2

# 参数设置完毕，开始训练模型
print("==================开始训练模型==================")
start_time = time.clock()
FRmodel.fit([lhs,rhs], labels, validation_split=0.2, batch_size=32, epochs=epoch, verbose=1)
end_time = time.clock()
mi = end_time-start_time
print("模型训练完毕！共训练了" + str(int(mi / 60)) + "分" + str(int(mi%60)) + "秒")

print("是否保存模型？[y/n]")
is_save = input()
if is_save == 'y':
    save_path = './weight/my_weight'
    FRmodel.save_weights(save_path)
    # 可以在设置模型参数后通过load_weight(file_name)方法加载权重
    print("保存成功！")