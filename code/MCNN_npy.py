#!/usr/bin/env python
# coding: utf-8

# # Dependency

# In[1]:


import h5py
import os
import pickle

from tqdm import tqdm
from time import gmtime, strftime


import numpy as np
import math


from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import tensorflow as tf
from tensorflow.keras import layers,Model

##

from sklearn.model_selection import KFold

import gc

import time
from sklearn.model_selection import KFold

#import import_test_ETC as load_data
import import_test as load_data
#import import_test_nine as load_data


# # PARAM

# In[2]:


DATA_LABEL=load_data.data_label()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n_fil","--num_filter", type=int, default=512, help="the number of filters in the convolutional layer")
parser.add_argument("-n_hid","--num_hidden", type=int, default=1000, help="the number of hidden units in the dense layer")
parser.add_argument("-bs","--batch_size", type=int, default=1024, help="the batch size")
parser.add_argument("-ws","--window_sizes", nargs="+", type=int, default=[2,4,6,8,10], help="the window sizes for convolutional filters")
parser.add_argument("-vm","--validation_mode", type=str, default="cross")
parser.add_argument("-d","--data_type", type=str,default="prottrans", help='"tape" "ProtTrans" "esm2" "esm15" ')
parser.add_argument("-n_feat","--num_feature", type=int, default=1024, help="the number of features")
parser.add_argument("-length","--n_length", type=int, default=5, help="the number of length")



args=parser.parse_args()



#DATA_TYPE = "ProtTrans"
#"/BinaryMatrix" "/MMseqs2" "/"
#"/tape" "/ProtTrans" "/esm"  

#"20" "20" 
#"1024" "768" "1280"
#NUM_FEATURE = 1024
NUM_HIDDEN = 1000#100

DATA_TYPE = args.data_type
NUM_FEATURE  = args.num_feature
NUM_FILTER = args.num_filter
BATCH_SIZE  = args.batch_size
WINDOW_SIZES = args.window_sizes
WINDOW = args.n_length
MAXSEQ = WINDOW*2+1

DATASET="cd_30"

NUM_CLASSES = 2
CLASS_NAMES = ['Negative','Positive']


EPOCHS      = 20

K_Fold = 5
VALIDATION_MODE=args.validation_mode
#"independent" "cross""LOAD"



#IMBLANCE="RANDOM"#"ADASYN" "RANDOM" "None" "SMOTE"
#SHUFFLE="Non-SHUFFLE"#Non-SHUFFLE


# In[3]:


import datetime

write_data=[]
a=datetime.datetime.now()
write_data.append(time.ctime())
write_data.append(DATA_LABEL)
write_data.append(DATA_TYPE)
write_data.append(BATCH_SIZE)
write_data.append(NUM_HIDDEN)
write_data.append(WINDOW_SIZES)
write_data.append(NUM_FILTER)
write_data.append(VALIDATION_MODE)

#write_data.append(IMBLANCE)
#write_data.append(SHUFFLE)


# # Time_log

# In[4]:


def time_log(message):
    print(message," : ",strftime("%Y-%m-%d %H:%M:%S", gmtime()))


# In[5]:


import time
import math
def SAVEROC(fpr,tpr,AUC):
    data_to_save = {
        "fpr": fpr,
        "tpr": tpr,
        "AUC": AUC
    }
    
    with open("./PKL/MCNN_"+str(math.floor(time.time()))+".pkl", "wb") as file:
        pickle.dump(data_to_save, file)


# In[6]:


"----------------------------------------------------------------------------------------------------"
# model fit batch funtion
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data))

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        return np.array(batch_data), np.array(batch_labels)
    
"----------------------------------------------------------------------------------------------------"


# # MCNN

# In[7]:


class DeepScan(Model):

	def __init__(self,
	             input_shape=(1, MAXSEQ, NUM_FEATURE),
	             window_sizes=[32],
	             num_filters=256,
	             num_hidden=1000):
		super(DeepScan, self).__init__()
		# Add input layer
		self.input_layer = tf.keras.Input(input_shape)
		self.window_sizes = window_sizes
		self.conv2d = []
		self.maxpool = []
		self.flatten = []
		for window_size in self.window_sizes:
			self.conv2d.append(
                layers.Conv2D(filters=num_filters,
    			              kernel_size=(1, window_size),
			                  activation=tf.nn.relu,
			                  padding='valid',
			                  bias_initializer=tf.constant_initializer(0.1),
			                  kernel_initializer=tf.keras.initializers.GlorotUniform()
                             )
                             )
			self.maxpool.append(
    			layers.MaxPooling2D(pool_size=(1, MAXSEQ - window_size + 1),
			                        strides=(1, MAXSEQ),
			                        padding='valid')
                               )
			self.flatten.append(layers.Flatten())
		self.dropout = layers.Dropout(rate=0.7)
		self.fc1 = layers.Dense(num_hidden,
                        		activation=tf.nn.relu,
		                        bias_initializer=tf.constant_initializer(0.1),
		                        kernel_initializer=tf.keras.initializers.GlorotUniform()
                               )
		self.fc2 = layers.Dense(NUM_CLASSES,
		                        activation='softmax',
		                        kernel_regularizer=tf.keras.regularizers.l2(1e-3)
                               )

		# Get output layer with `call` method
		self.out = self.call(self.input_layer)

	def call(self, x, training=False):
		_x = []
		for i in range(len(self.window_sizes)):
			x_conv = self.conv2d[i](x)
			x_maxp = self.maxpool[i](x_conv)
			x_flat = self.flatten[i](x_maxp)
			_x.append(x_flat)

		x = tf.concat(_x, 1)
		x = self.dropout(x, training=training)
		x = self.fc1(x)
		x = self.fc2(x)  #Best Threshold
		return x


# # Main

# In[8]:


# Example usage:

x_train,y_train,x_test,y_test= load_data.MCNN_data_load(DATA_TYPE)

# In[9]:


print(x_train.shape)
print(x_train.dtype)
print(y_train.shape)
print()
print(x_test.shape)
print(x_test.dtype)
print(y_test.shape)


# In[10]:





# In[11]:


def model_test(model, x_test, y_test):

    #print(x_test.shape)
    pred_test = model.predict(x_test)
    #print(pred_test)
    #print(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    ROCAUC = metrics.auc(fpr, tpr)
    #tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

    precisions, recalls, threshold = precision_recall_curve(y_test[:,1], pred_test[:, 1])
    PRAUC = metrics.auc(recalls, precisions)
    
    
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=ROCAUC, estimator_name='mCNN')
    display.plot()

    disp2 =metrics.PrecisionRecallDisplay(precision= precisions, recall=recalls,average_precision=PRAUC,estimator_name='mCNN')
    disp2.plot()
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]

    y_pred = (pred_test[:, 1] >= threshold).astype(int)
    #print(y_pred)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
    F1 = 2*TP/(2*TP+FP+FN)
    Prec=TP/(TP+FP)
    Recall=TP/(TP+FN)
    print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, ROCAUC={ROCAUC:.4f}, F1={F1:.4f}, Prec={Prec:.4f}, Recall={Recall:.4f}, PRAUC={PRAUC:.4f}\n')
    #SAVEROC(fpr,tpr,AUC)
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,ROCAUC,F1,Prec,Recall,PRAUC


# In[12]:


if(VALIDATION_MODE=="cross"):
	time_log("Start cross")
	
	kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
	results=[]
	i=1
	for train_index, test_index in kfold.split(x_train):
		print(i,"/",K_Fold,'\n')
		# 取得訓練和測試數據
		X_train, X_test = x_train[train_index], x_train[test_index]
		Y_train, Y_test = y_train[train_index], y_train[test_index]
		
		print(X_train.shape)
		print(X_test.shape)
		print(Y_train.shape)
		print(Y_test.shape)
		#generator = DataGenerator(X_train, Y_train, batch_size=BATCH_SIZE)
		# 重新建模
		model = DeepScan(
		num_filters=NUM_FILTER,
			num_hidden=NUM_HIDDEN,
			window_sizes=WINDOW_SIZES)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.build(input_shape=X_train.shape)
		model.summary()
		# 在測試數據上評估模型
		history=model.fit(
			X_train,
            Y_train,
			epochs=EPOCHS,
			callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)],
			verbose=1,
			shuffle=True
		)
		TP,FP,TN,FN,Sens,Spec,Acc,MCC,ROCAUC,F1,Prec,Recall,PRAUC = model_test(model, X_test, Y_test)
		results.append([TP,FP,TN,FN,Sens,Spec,Acc,MCC,ROCAUC,F1,Prec,Recall,PRAUC])
		i+=1
		
		del X_train
		del X_test
		del Y_train
		del Y_test
		gc.collect()
		
	mean_results = np.mean(results, axis=0)
	print(f'TP={mean_results[0]:.4}, FP={mean_results[1]:.4}, TN={mean_results[2]:.4}, FN={mean_results[3]:.4}, Sens={mean_results[4]:.4}, Spec={mean_results[5]:.4}, Acc={mean_results[6]:.4},MCC={mean_results[7]:.4},ROCAUC={mean_results[8]:.4},F1={mean_results[9]:.4},Prec={mean_results[10]:.4},Recall={mean_results[11]:.4},PRAUC={mean_results[12]:.4}\n')
    
	write_data.append(mean_results[0])
	write_data.append(mean_results[1])
	write_data.append(mean_results[2])
	write_data.append(mean_results[3])
	write_data.append(mean_results[4])
	write_data.append(mean_results[5])
	write_data.append(mean_results[6])
	write_data.append(mean_results[7])
	write_data.append(mean_results[8])
	write_data.append(mean_results[9])
	write_data.append(mean_results[10])
	write_data.append(mean_results[11])
	write_data.append(mean_results[12])





# In[13]:


if(VALIDATION_MODE=="independent"):
    generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
    time_log("Start Model Train")
    model = DeepScan(
    	num_filters=NUM_FILTER,
    	num_hidden=NUM_HIDDEN,
    	window_sizes=WINDOW_SIZES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=x_train.shape)
    model.summary()
    model.fit(
    	x_train,
        y_train,
    	epochs=EPOCHS,
    	shuffle=True,
    )
    time_log("End Model Train")
    time_log("Start Model Test")
    TP,FP,TN,FN,Sens,Spec,Acc,MCC,ROCAUC,F1,Prec,Recall,PRAUC = model_test(model, x_test, y_test)
    write_data.append(TP)
    write_data.append(FP)
    write_data.append(TN)
    write_data.append(FN)
    write_data.append(Sens)
    write_data.append(Spec)
    write_data.append(Acc)
    write_data.append(MCC)
    write_data.append(ROCAUC)
    write_data.append(F1)
    write_data.append(Prec)
    write_data.append(Recall)
    write_data.append(PRAUC)
    
    time_log("End Model Test")


# In[14]:


def save_csv(write_data,a):
    import csv
    b=datetime.datetime.now()
    write_data.append(b-a)
    open_csv=open("./single.csv","a")
    write_csv=csv.writer(open_csv)
    write_csv.writerow(write_data)
save_csv(write_data,a)


# In[ ]:




