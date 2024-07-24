from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc

datalabel="ca2+channels"

def data_label():
    return datalabel

#def MCNN_data_load(MAXSEQ):
def MCNN_data_load(DATA_TYPE):

    #path_m_training = "../dataset/"+str(MAXSEQ)+"/mcarrier/train.npy"
    #path_s_training = "../dataset/"+str(MAXSEQ)+"/secondary/train.npy"
    
    path_data_training = "../dataset/"+str(DATA_TYPE)+"/train/data.npy"
    path_label_training = "../dataset/"+str(DATA_TYPE)+"/train/label.npy"

    path_data_testing = "../dataset/"+str(DATA_TYPE)+"/test/data.npy"
    path_label_testing = "../dataset/"+str(DATA_TYPE)+"/test/label.npy"
    
    x_train,y_train=data_load(path_data_training,path_label_training)
    x_test,y_test=data_load(path_data_testing,path_label_testing)
    
    return(x_train,y_train,x_test,y_test)

def data_load(DATA,LABEL):
    data=np.load(DATA)
    label=np.load(LABEL)
    
    #label1 = np.ones(f1.shape[0])
    #label2 = np.zeros(f2.shape[0])
   
    
    #print(data.shape)
    #print(label.shape)
    
    #print(label1)
    #print(label2)
  
    
    x=data
    y= tf.keras.utils.to_categorical(label,2)
    #y.dtype='float16'
    gc.collect()
    return x ,y