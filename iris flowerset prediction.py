#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# # Data Collection & pre-processing

# In[8]:


iris_data = pd.read_csv('D:\ML Files\IRIS Flowerset Prediction\iris.csv',names=['Sep len','Sep wid','Pet len','Pet wid','Class'],header=None)


# In[9]:


iris_data.head()


# Attribute Information:
#    1. sepal length in cm
#    2. sepal width in cm
#    3. petal length in cm
#    4. petal width in cm
#    5. class: 
#       -- Iris Setosa
#       -- Iris Versicolour
#       -- Iris Virginica

# In[10]:


iris_data.shape


# In[11]:


iris_data.describe()


# In[7]:


iris_data.isnull().sum()


# In[16]:


iris_data.groupby('Class').count()


# In[17]:


#Replacing class by numeric value
iris_data.replace({'Class':{'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}},inplace=True)


# Iris-setosa':1
# 'Iris-versicolor':2
# 'Iris-virginica':3

# In[20]:


iris_data.head()


# In[21]:


iris_data.tail()


# # Splitting dataset into features and labels

# In[24]:


X = iris_data.drop(columns='Class',axis=1)
Y = iris_data['Class']


# In[25]:


print(X)


# print(Y)

# # Spliiting features and labels for train & test

# In[27]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify=Y,test_size=0.2,random_state=5)


# In[28]:


print(X.shape,X_train.shape,X_test.shape)


# # Model Training

# In[35]:


model = LogisticRegression()


# In[36]:


model.fit(X_train,Y_train)


# # Model Evaluation

# In[38]:


#Accuracy Score for training data
train_prediction = model.predict(X_train)
acc_score_train = accuracy_score(train_prediction,Y_train)
print("Accuracy Score for training dataset : ",round(acc_score_train*100,2))


# In[39]:


#Accuracy Score for testing data
test_prediction = model.predict(X_test)
acc_score_test = accuracy_score(test_prediction,Y_test)
print("Accuracy Score for testing dataset : ",round(acc_score_test*100,2))


# # Building a predictive system

# In[42]:


input_data = (6.3,2.9,5.6,1.8)

#5.3,3.7,1.5,0.2,Iris-setosa
#5.0,3.3,1.4,0.2,Iris-setosa
#7.0,3.2,4.7,1.4,Iris-versicolor
#6.4,3.2,4.5,1.5,Iris-versicolor
#7.1,3.0,5.9,2.1,Iris-virginica
#6.3,2.9,5.6,1.8,Iris-virginica

input_np = np.asarray(input_data)

input_res = input_np.reshape(1,-1)

predicted = model.predict(input_res)

if(predicted[0] == 1):
    print("Iris-setosa")
elif(predicted[0] == 2):
    print("Iris-versicolor")
elif(predicted[0] == 3):
    print("Iris-virginica")


# In[ ]:




