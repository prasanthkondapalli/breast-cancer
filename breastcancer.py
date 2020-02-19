# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:05:04 2020

@author: Prasanth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('F:/ASS/BREAST_CANCER')
print(df.columns)


df.isnull().sum() 


x=df.copy()

del x['diagnosis (Target)']
del x['id']

y=df['diagnosis (Target)']

plt.boxplot(x['radius_mean'])
plt.boxplot(x['texture_mean'])
plt.boxplot(x['perimeter_mean'])
plt.boxplot(x['area_mean'])
plt.boxplot(x['smoothness_mean'])
plt.boxplot(x['compactness_mean'])
plt.boxplot(x['concavity_mean'])
plt.boxplot(x['points_mean'])
plt.boxplot(x['symmetry_mean'])
plt.boxplot(x['dimension_mean'])
plt.boxplot(x['radius_se'])
plt.boxplot(x['texture_se'])
plt.boxplot(x['perimeter_se'])
plt.boxplot(x['area_se'])
plt.boxplot(x['smoothness_se'])
plt.boxplot(x['compactness_se'])
plt.boxplot(x['concavity_se'])
plt.boxplot(x['points_se'])
plt.boxplot(x['symmetry_se'])
plt.boxplot(x['dimension_se'])
plt.boxplot(x['radius_worst'])
plt.boxplot(x['texture_worst'])
plt.boxplot(x['perimeter_worst'])
plt.boxplot(x['area_worst'])
plt.boxplot(x['smoothness_worst'])
plt.boxplot(x['compactness_worst'])
plt.boxplot(x['concavity_worst'])
plt.boxplot(x['points_worst'])
plt.boxplot(x['symmetry_worst'])
plt.boxplot(x['dimension_worst'])



per=x['radius_mean'].quantile([0.0,0.95]).values
x['radius_mean']=x['radius_mean'].clip(per[0],per[1])

per=x['texture_mean'].quantile([0,0.97]).values
x['texture_mean']=x['texture_mean'].clip(per[0],per[1])

per=x['perimeter_mean'].quantile([0,0.97]).values
x['perimeter_mean']=x['perimeter_mean'].clip(per[0],per[1])

per=x['area_mean'].quantile([0,0.95]).values
x['area_mean']=x['area_mean'].clip(per[0],per[1])

per=x['smoothness_mean'].quantile([0.1,0.97]).values
x['smoothness_mean']=x['smoothness_mean'].clip(per[0],per[1])

per=x['compactness_mean'].quantile([0,0.97]).values
x['compactness_mean']=x['compactness_mean'].clip(per[0],per[1])

per=x['concavity_mean'].quantile([0.1,0.967]).values
x['concavity_mean']=x['concavity_mean'].clip(per[0],per[1])

per=x['points_mean'].quantile([0,0.97]).values
x['points_mean']=x['points_mean'].clip(per[0],per[1])

per=x['symmetry_mean'].quantile([0.1,0.97]).values
x['symmetry_mean']=x['symmetry_mean'].clip(per[0],per[1])

per=x['dimension_mean'].quantile([0,0.97]).values
x['dimension_mean']=x['dimension_mean'].clip(per[0],per[1])

per=x['radius_se'].quantile([0,0.934]).values
x['radius_se']=x['radius_se'].clip(per[0],per[1])


per=x['texture_se'].quantile([0,0.964]).values
x['texture_se']=x['texture_se'].clip(per[0],per[1])


per=x['perimeter_se'].quantile([0,0.934]).values
x['perimeter_se']=x['perimeter_se'].clip(per[0],per[1])


per=x['area_se'].quantile([0,0.885]).values
x['area_se']=x['area_se'].clip(per[0],per[1])


per=x['smoothness_se'].quantile([0,0.934]).values
x['smoothness_se']=x['smoothness_se'].clip(per[0],per[1])


per=x['compactness_se'].quantile([0,0.94567]).values
x['compactness_se']=x['compactness_se'].clip(per[0],per[1])


per=x['concavity_se'].quantile([0,0.94567]).values
x['concavity_se']=x['concavity_se'].clip(per[0],per[1])


per=x['points_se'].quantile([0,0.967]).values
x['points_se']=x['points_se'].clip(per[0],per[1])


per=x['symmetry_se'].quantile([0,0.947]).values
x['symmetry_se']=x['symmetry_se'].clip(per[0],per[1])


per=x['dimension_se'].quantile([0,0.947]).values
x['dimension_se']=x['dimension_se'].clip(per[0],per[1])


per=x['radius_worst'].quantile([0,0.97]).values
x['radius_worst']=x['radius_worst'].clip(per[0],per[1])


per=x['texture_worst'].quantile([0,0.987]).values
x['texture_worst']=x['texture_worst'].clip(per[0],per[1])


per=x['perimeter_worst'].quantile([0,0.97]).values
x['perimeter_worst']=x['perimeter_worst'].clip(per[0],per[1])

per=x['area_worst'].quantile([0,0.934567]).values
x['area_worst']=x['area_worst'].clip(per[0],per[1])


per=x['smoothness_worst'].quantile([0.1,0.987]).values
x['smoothness_worst']=x['smoothness_worst'].clip(per[0],per[1])


per=x['concavity_worst'].quantile([0,0.978]).values
x['concavity_worst']=x['concavity_worst'].clip(per[0],per[1])


per=x['symmetry_worst'].quantile([0,0.96]).values
x['symmetry_worst']=x['symmetry_worst'].clip(per[0],per[1])

per=x['dimension_worst'].quantile([0,0.9589]).values
x['dimension_worst']=x['dimension_worst'].clip(per[0],per[1])

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)





from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(xtrain,ytrain)

ypred_ts=reg.predict(xtest)
ypred_tr=reg.predict(xtrain)

print(accuracy_score(ytest,ypred_ts))#98
print(accuracy_score(ytrain,ypred_tr))#99


print(confusion_matrix(ytest.values,ypred_ts))
print(confusion_matrix(ytrain.values,ypred_tr))

from sklearn.metrics import classification_report
print(classification_report(ytrain,ypred_tr))
print(classification_report(ytest,ypred_ts))

print(reg.coef_)
print(reg.score)







