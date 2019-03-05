# -*- coding: utfq-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image
#PIL是python映像库
import os

image_train_path = './cifar-10/cifar-10/train/'
label_train_path = './'
tfRecord_train = './cifar-10_tfRecord/train/'
image_test_path = './cifar-10/cifar-10/test/'
label_test_path = './'
tfRecord_test = './cifar-10_tfRecord/test/'
data_path = '/cifar_data'
resize_height = 32
resize_width = 32
#TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议
#内存块包含了字段 Features)。我们可以写一段代码获取你的数据， 将数据填入
#到Example协议内存块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过
#tf.python_io.TFRecordWriter 写入到TFRecords文件。

def write_tfRecord(tfRecordName,image_path,label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
#    f = open(label_path,'r')
#    ###打开一个标签txt文件，每行由文件名及标签组成
#    contents = f.readlines()
#    ##readlines文件返回一个列表
#    f.close()
    with open(label_path) as f:
        contents = f.readlines()
        #label_path是标签文件（应该是自定义的）的存储路径
#        比如下面的文件
#        test_array.txt
    #img_1.jpg 1
    #img_2.jpg 2
    #img_3.jpg 3
    #img_4.jpg 4
    #img_5.jpg 5
    #img_6.jpg 6
    #img_7.jpg 7
    #img_8.jpg 8
    #img_9.jpg 9
    for content in contents:
        value = content.split()#所有空格换行制表符
        img_path = image_path + value[0]
        img = Image.open(img_path)
#        返回一个图像对象
        img_raw = img.tobytes()
        labels = [0] * 10
        labels[int(value[1])] = 1
        
        example = tf.train.Example(features=tf.train.Features(feature={'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                                                                       'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}))
#        基本的，一个Example中包含Features，Features里
#        包含Feature（这里没s）的字典。最后，Feature里包
#        含有一个 FloatList， 或者ByteList，或者Int64List
        writer.write(example.SerializeToString())
        num_pic += 1
        print ("the number of picture",num_pic)
    writer.close()
    print("write tfrecord successfully")
    
        
def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("the directory was created successfully")
    else:
        print("direcory existed")
    write_tfRecord(tfRecord_train,image_train_path,label_train_path)
    write_tfRecord(tfRecord_test,image_test_path,label_test_path)
    

def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example, features={'label':tf.FixedLenFeature([10],tf.int64),
                                                                     'img_raw':tf.FixedLenFeature([],tf.string)})
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img.set_shape([784])
    img = tf.cast(img,tf.float32) * (1./255)
    label = tf.cast(features['label'],tf.float32)
    return img,label

def get_tfrecord(num,isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img,label = read_tfRecord(tfRecord_path)
    img_batch,label_batch = tf.train.shuffle_batch([img,label],
                                                   batch_size=num,
                                                   num_threads = 2,
                                                   capacity = 1000,
                                                   min_after_dequeue=700)
#    读取一个文件并且加载一个张量中的batch_size行
    return img_batch,label_batch

def main():
    generate_tfRecord()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    