#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"D:\data science\student\student_scores.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np


# In[7]:


df.plot.scatter(x='Hours',y='Scores',title='scatter plot for student score vs hours')


# In[8]:


print(df.corr())


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


y=df['Scores'].values.reshape(-1,1)


# In[11]:


y


# In[12]:


x=df['Hours'].values.reshape(-1,1)


# In[13]:


x


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[17]:


y_train.shape


# In[18]:


y_test.shape


# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


reg=LinearRegression()


# In[21]:


reg.fit(x_train,y_train)


# In[22]:


reg.intercept_


# In[23]:


reg.coef_


# In[24]:


def calc(slope,intercept,hours):
    return slope*hours+intercept


# In[25]:


slope=calc(reg.coef_,reg.intercept_,9.5)


# In[26]:


print(slope)


# In[27]:


y_pred=reg.predict(x_test)


# In[28]:


y_pred


# In[29]:


df_pred=pd.DataFrame({'Hours':x_test.squeeze(),'Actual':y_test.squeeze(),'predict':y_pred.squeeze()})


# In[30]:


df_pred


# In[31]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[32]:


mae=mean_absolute_error(y_test,y_pred)


# In[33]:


mae


# In[34]:


mse=mean_squared_error(y_test,y_pred)


# In[35]:


mse


# In[36]:


rmse=np.sqrt(mse)


# In[37]:


rmse


# In[ ]:




