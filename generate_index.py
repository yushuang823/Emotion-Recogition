# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:02:50 2019

@author: MR toad
"""

#generating the label(index)

import os

RAW_FILE_DIR = "./cifar-10_raw/cifar-10/"

def generate_index(raw_file_dir, USAGE):
    #利用listdir可以获取文件夹中所有文件的名称
    usage = USAGE
    if usage == 'train':
        f = open("train_index.txt",'a+')
        real_file_entrance = raw_file_dir + 'train/'
        typename_list = os.listdir(real_file_entrance)
        for typename in typename_list:
            file_name_list = os.listdir(real_file_entrance + typename + '/')
            for i in range(len(file_name_list)):
                file_name_list[i] = file_name_list[i] + ' ' + typename + '\n'
            f.writelines(file_name_list)
        f.close()
    elif usage == 'test':
        f = open("test_index.txt",'a+')
        real_file_entrance = raw_file_dir + 'test/'
        typename_list = os.listdir(real_file_entrance)
        for typename in typename_list:
            file_name_list = os.listdir(real_file_entrance + typename + '/')
            for i in range(len(file_name_list)):
                file_name_list[i] = file_name_list[i] + ' ' + typename + '\n'
            f.writelines(file_name_list)
        f.close()
    else:
        print("wrong directory")
        return

def checkfile(usage):
    if usage == 'test':
        return os.path.exists("test_index.txt")
    elif usage == 'train':
        return os.path.exists("train_index.txt")
    
    
def main():
    if checkfile('test') == False:
        generate_index(RAW_FILE_DIR,'test')
    if checkfile('train') == False:
        generate_index(RAW_FILE_DIR,'train')
    
if __name__ == '__main__':
    main()
        
        
    
                
                
                
            