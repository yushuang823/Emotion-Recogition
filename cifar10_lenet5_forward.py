# -*- coding: utf-8 -*-


import tensorflow as tf
IMAGE_SIZE = 32#图片大小
NUM_CHANNELS = 3#图片通道数，由于是识别三种颜色，通道数是三
CONV1_SIZE = 5#第一层卷积核的大小
CONV1_KERNEL_NUM = 32#第一层卷积核的数量，其值作为卷积后输出的图片的层数
CONV2_SIZE = 5#第二层卷积核的大小
CONV2_KERNEL_NUM = 64#第二层卷积核的数量，其值作为卷积后输出的图片的层数
FC_SIZE = 512#全连接神经网络（隐藏层神经元）的数量
OUTPUT_NODE = 10#输出层神经元换的数量

def get_weight(shape,regularizer):#生成权重的函数，需要形状及正则化参数
    w = tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))
    #按照舍弃过大偏离的高斯分布随机生成参数
    if regularizer != None:
        tf.add_to_collection('losses',
            tf.contrib.layers.l2_regularizer(regularizer)(w))
        #采用l2方法计算计算正则化，并把得到的值加入集合‘losses’中
    return w

def get_bias(shape):#生成偏置的函数
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#进行卷积运算的函数，其中x已经reshape成4维张量，w是卷积核中的偏重，采用全零填充
#注意x,w中各维数的含义
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train,regularizer):
    conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x,conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    pool1 = max_pool_2x2(relu1)
    
    conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1,conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    pool2 = max_pool_2x2(relu2)
    
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
    
    fc1_w = get_weight([nodes,FC_SIZE],regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1,0.5)
    
    fc2_w = get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1,fc2_w) + fc2_b
    return y