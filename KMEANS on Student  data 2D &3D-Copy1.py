#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


df = pd.read_csv('STUDENT_CGPA.csv')
df.shape


# In[5]:


df


# In[6]:


df.head(5)


# In[7]:


import matplotlib.pyplot as plt
plt.scatter(df['CGPA'],df['IQ LEVEL'])


# In[8]:


from sklearn.cluster import KMeans


# In[9]:


stu = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(df)
    stu.append(km.inertia_)


# In[10]:


stu


# In[11]:


plt.plot(range(1,11),stu)


# In[12]:


X = df.iloc[:,:].values
km = KMeans(n_clusters=4)
y_means = km.fit_predict(X)


# In[13]:


y_means


# In[14]:


X[y_means == 3,1]


# In[15]:


plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='blue')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='red')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color='yellow')


# # KMEANS ON 3D

# In[16]:


from sklearn.datasets import make_blobs

centroids = [(-5,-5,5),(5,5,-5),(3.5,-2.5,4),(-2.5,2.5,-4)]
cluster_std = [1,1,1,1]

X,y = make_blobs(n_samples=200,cluster_std=cluster_std,centers=centroids,n_features=3,random_state=1)


# In[17]:


X


# In[18]:


import plotly.express as px
fig = px.scatter_3d(x=X[:,0], y=X[:,1], z=X[:,2])
fig.show()


# In[19]:


stu = []
for i in range(1,21):
    km = KMeans(n_clusters=i)
    km.fit_predict(X)
    stu.append(km.inertia_)


# In[20]:


plt.plot(range(1,21),stu)


# In[21]:


km = KMeans(n_clusters=4)
y_pred = km.fit_predict(X)


# In[22]:


df = pd.DataFrame()
df['col1'] = X[:,0]
df['col2'] = X[:,1]
df['col3'] = X[:,2]
df['label'] = y_pred


# In[23]:


fig = px.scatter_3d(df,x='col1', y='col2', z='col3',color='label')
fig.show()


# In[ ]:





# In[ ]:




