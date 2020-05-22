# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:16:39 2020

@author: SURYANSH
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset =pd.read_csv('kidney_disease.csv')
dataset =dataset.drop('id',1)
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 24].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN' , strategy ='mean' , axis= 0)
imputer.fit(x[:, 0:3])
x[:, 0:3] = imputer.transform(x[:, 0:3])
imputer1 = Imputer(missing_values='NaN' , strategy='mean' , axis=0)
imputer1.fit(x[:, 9:18])
x[:, 9:18]=imputer1.transform(x[:, 9:18])
from sklearn_pandas import CategoricalImputer
imputer2=CategoricalImputer()
x[:, 3:9]=imputer2.fit_transform(x[:, 3:9])


