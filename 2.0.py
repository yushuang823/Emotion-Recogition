# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:06:47 2019

@author: 93668
"""
import os
import scipy
import speechpy
import sklearn
import numpy
mfcc_len=39
def read_wav(filename):
        return scipy.io.wavfile.read(filename)
def enframe(wavData, frameSize, overlap):
    coeff = 0.97 # 预加重系数
    wlen = len(wavData)
    step = frameSize - overlap
    frameNum:int = math.ceil(wlen / step)
    frameData = np.zeros((frameSize, frameNum))

    hamwin = np.hamming(frameSize)

    for i in range(frameNum):
        singleFrame = wavData[np.arange(i * step, min(i * step + frameSize, wlen))]
        singleFrame = np.append(singleFrame[0], singleFrame[:-1] - coeff * singleFrame[1:]) # 预加重
        frameData[:len(singleFrame), i] = singleFrame
        frameData[:, i] = hamwin * frameData[:, i] # 加窗
        return frameData
def read():
    dataset_folder="dataset/"
    class_labels={"Neutral":0,"Angry":2,"Happy":3,"sad":1}
    for i, directory in enumerate(class_labels):
        print("started reading folders",directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal=read_wav(filename)
            signal=enframe(signal,10,1)
            mfcc=speechpy.feature.mfcc(signal,fs,num_cepstral=mfcc_len)
            x_train, x_test, y_train, y_test=sklearn.model_selection.train_test_split(dataset_folder,class_labels)
            print (type(fs))
    return numpy.array(x_train),numpy.array(x_test),numpy(y_train),numpy(y_test)
