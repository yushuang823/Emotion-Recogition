# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:41:08 2019

@author: MR toad
"""

import librosa
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
#from PIL import Image

file_path = 'E:/speech/'
mfcc_path = 'E:/mfcc/'
pic_path = 'E:/pic'

file_name_list = os.listdir(file_path)
for file_name in file_name_list:
    y, sr = librosa.load(file_path+file_name)
    mfcc_feature = librosa.feature.mfcc(y=y, sr=sr)
    np.save(mfcc_path+file_name.split('.')[0]+".npy",mfcc_feature)
    
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.savefig(file_name.split('.')[0]+".png",dpi=300)
    


    