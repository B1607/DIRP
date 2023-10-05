#!/usr/bin/env python
# coding: utf-8

# # Dependency

# In[1]:


import numpy as np
import math

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold


import gc

import loading_function as load_data
import MCNN 


# # PARAM

# In[2]:



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_type", type=str,default="ProtTrans", help='"BinaryMatrix" "MMseqs2" "ProtTrans"')
parser.add_argument("-n_dep","--num_dependent", type=int, default=5, help="the number of dependent variables")
parser.add_argument("-n_fil","--num_filter", type=int, default=256, help="the number of filters in the convolutional layer")
parser.add_argument("-n_hid","--num_hidden", type=int, default=1000, help="the number of hidden units in the dense layer")
parser.add_argument("-bs","--batch_size", type=int, default=1024, help="the batch size")
parser.add_argument("-ws","--window_sizes", nargs="+", type=int, default=[2,4,6,8,10], help="the window sizes for convolutional filters")
parser.add_argument("-n_feat","--num_feature", type=int, default=1024, help="the number of features")
parser.add_argument("-e","--epochs", type=int, default=20, help="the number of epochs for training")
parser.add_argument("-val","--validation_mod", type=str, default="independent", help="the mod for validation 'cross' or 'independent'")



args=parser.parse_args()
NUM_DEPENDENT =args.num_dependent
MAXSEQ = NUM_DEPENDENT*2+1

DATA_TYPE = args.data_type
#"/BinaryMatrix" "/MMseqs2" "/ProtTrans"


NUM_FILTER = args.num_filter
NUM_HIDDEN =args.num_hidden
BATCH_SIZE  = args.batch_size
WINDOW_SIZES = args.window_sizes




NUM_CLASSES = 2

NUM_FEATURE = args.num_feature
EPOCHS      = args.epochs
K_Fold =5
VALIDATION_MODE=args.validation_mod
#"independent" "cross"



# # Main

# In[3]:


# Example usage:
x_train, y_train,x_test, y_test = load_data.MCNN_data_load(DATA_TYPE,NUM_CLASSES,NUM_DEPENDENT)


# In[4]:


def model_test(model, x_test, y_test):
    
    print(x_test.shape)
    pred_test = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test[:,1], pred_test[:, 1])
    AUC = metrics.auc(fpr, tpr)

    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    #print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
    threshold = thresholds[ix]
    y_pred = (pred_test[:, 1] >= threshold).astype(int)

    TN, FP, FN, TP =  metrics.confusion_matrix(y_test[0:][:,1], y_pred).ravel()

    Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
    Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
    Acc = (TP+TN)/(TP+FP+TN+FN)
    MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
   
    return TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC,fpr,tpr,thresholds[ix],gmeans[ix]


# In[5]:


if(VALIDATION_MODE=="cross"):
	
	kfold = KFold(n_splits = K_Fold, shuffle = True, random_state = 2)
	results=[]
	i=1
	for train_index, test_index in kfold.split(x_train):
		print(i,"/",K_Fold,'\n')
	
		X_train, X_test = x_train[train_index], x_train[test_index]
		Y_train, Y_test = y_train[train_index], y_train[test_index]
	
		model = MCNN.DeepScan(
            num_class=NUM_CLASSES,
            maxseq=MAXSEQ,
            input_shape=(1, MAXSEQ, NUM_FEATURE),
    		num_filters=NUM_FILTER,
			num_hidden=NUM_HIDDEN,
			window_sizes=WINDOW_SIZES)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.build(input_shape=X_train.shape)
		model.summary()
		history=model.fit(
			X_train,
			Y_train,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			verbose=0,
			shuffle=True
		)
		TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC,fpr,tpr,thresholds,gmeans = model_test(model, X_test, Y_test)
		results.append([Sens, Spec, Acc, MCC, AUC])
		i+=1
		
		del X_train
		del X_test
		del Y_train
		del Y_test
		gc.collect()
		
	mean_results = np.mean(results, axis=0)
	print(f'Sens={mean_results[0]:.4}, Spec={mean_results[1]:.4}, Acc={mean_results[2]:.4}, MCC={mean_results[3]:.4}, AUC={mean_results[4]:.4}\n')


# In[7]:


if(VALIDATION_MODE=="independent"):
	model = MCNN.DeepScan(
        num_class=NUM_CLASSES,
            maxseq=MAXSEQ,
            input_shape=(1, MAXSEQ, NUM_FEATURE),
    		num_filters=NUM_FILTER,
			num_hidden=NUM_HIDDEN,
			window_sizes=WINDOW_SIZES)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.build(input_shape=x_train.shape)
	model.fit(
		x_train,
		y_train,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
        verbose=0,
		shuffle=True
    )
	TP,FP,TN,FN,Sens,Spec,Acc,MCC,AUC,fpr,tpr,thresholds,gmeans = model_test(model, x_test, y_test)
	display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=AUC, estimator_name='mCNN')
	display.plot()
	print(f'Best Threshold={thresholds}, G-Mean={gmeans}')
	print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, AUC={AUC:.4f}\n')


# In[ ]:





# In[ ]:




