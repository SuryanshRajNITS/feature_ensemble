# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:16:39 2020

@author: SURYANSH
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('SPECTF_HeartDataSet.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 44].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test= sc_x.transform(x_test)

#from sklearn.decomposition import PCA
#pca =PCA(n_components= None)
#x_train=pca.fit_transform(x_train)
#x_test=pca.transform(x_test)
#ev=pca.explained_variance_ratio_*/
from sklearn.decomposition import PCA
pca =PCA(n_components= 5)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
ev=pca.explained_variance_ratio_


