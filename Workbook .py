#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("USA_Housing.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[7]:


df.columns


# In[8]:


sns.pairplot(df)


# In[10]:


sns.displot(df["Price"])


# In[16]:


sns.heatmap(df.corr(),annot = True)


# In[17]:


df.columns


# In[21]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[22]:


X.head()


# In[25]:


y = df["Price"]


# In[24]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


lm = LinearRegression()


# In[30]:


lm.fit(X_train, y_train)


# In[31]:


print(lm.intercept_)


# In[33]:


lm.coef_


# In[34]:


X.columns


# In[35]:


cdf = pd.DataFrame(lm.coef_, X.columns, columns = ["Coeff"])


# In[37]:


cdf


# Predictions

# In[41]:


predictions = lm.predict(X_test) 


# In[42]:


predictions


# In[43]:


y_test


# In[44]:


plt.scatter(y_test,predictions)


# In[45]:


sns.displot(y_test - predictions)


# In[46]:


from sklearn import metrics


# In[47]:


metrics.mean_absolute_error(y_test, predictions)


# In[48]:


metrics.mean_squared_error(y_test, predictions)


# In[49]:


from math import sqrt 


# In[50]:


sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:




