# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:11:32 2019

@author: hemanth
"""

   
#from random import randint,uniform
#from numpy import mean,std
import pandas as pd
#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from numpy import var

'''
#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import  QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
'''

#Data input from the file
data=pd.read_csv("bitcoin_data_vol_7.csv")#,header= None)
X=data.drop(["Date","Open", "High", "Low"], axis=1)
#print data["Close"]
Y=data["Close"].shift(1)
#Y.Close=Y.Close
#print Y
Y[0]=data["Close"][0]
#print Y

n=int(0.9*data.__len__())
#tscv = TimeSeriesSplit(n_splits=int((len(Y)-3)/3))
#for train_index, test_index in tscv.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)

#To get the indices 
X_train, X_test = X.loc[:n], X.loc[n+1:]
Y_train, Y_test = Y.loc[:n], Y.loc[n+1:]

def calc_volatility(data, time):
    vol=[]
    for i in range(data.__len__()):
        #first case trivial - one element it will be 0
        if i == 0:
            vol.append(0)
        #less than the elements are available. So variance of fewer elements
        elif(i<time):
            vol.append(var(data[0:i+1]))
        #general case where we take last 7 days
        else:
            vol.append(var(data[i-time+1:i+1]))
    return vol

def conv_into_array(l, start):
    lis=[]    
    for i in range(l.__len__()):
        lis.append(l.loc[start+i])
    return lis

def calc_error(Y1,Y2):
    #calculate error between predicted y and test 
    err=0
    for i in range(Y1.__len__()):
        err+=abs(Y1[i]-Y2[i])
    return err

reg1 = GradientBoostingRegressor()
reg2 = RandomForestRegressor()

reg1.fit(X_train, Y_train)
reg2.fit(X_train, Y_train)

Y1=reg1.predict(X_test)
Y2=reg2.predict(X_test)
Y_corr=conv_into_array(Y_test,n+1)

print "Error from GBR, using 7day volatility",calc_error(Y_corr,Y1)
print "Error from RFR, using 7day volatility",calc_error(Y_corr,Y2)

fig = plt.figure()
plt.plot([x for x in range(Y_test.__len__())], Y_test, 'g-', label=r'Day vs Y_test')
plt.plot([x for x in range(Y1.__len__())], Y1, 'b-', label=u'Y1')
plt.plot([x for x in range(Y2.__len__())], Y2, 'r-', label=u'Y2')
plt.xlabel('$Day$')
plt.ylabel('$Price$')
plt.legend(loc='upper left')
plt.show()