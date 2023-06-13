#!/usr/bin/env python
# coding: utf-8

# In[1]:


#customer segmentation using k-means clustering 


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#lets load the data


# In[4]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\Mall_Customers.csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


#there is no wrong data,null values and duplicates in this data set


# In[11]:


#now lets do modelling 


# In[39]:


x = df.iloc[:,3:5].values


# In[ ]:





# # k-means clustering

# In[41]:


from sklearn.cluster import KMeans


# In[49]:


wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[50]:


print(wcss)


# In[51]:


plt.plot(range(1,11),wcss)
plt.xticks(range(1,11))
plt.title("Elbow method")
plt.xlabel("K-clusters values")
plt.ylabel("WCSS")
plt.show()


# In[52]:


kmeans = KMeans(n_clusters=5,init='k-means++')


# In[53]:


y_kmeans = kmeans.fit_predict(x)


# In[54]:


y_kmeans


# In[55]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],

s = 100, c = 'red', label = 'Cluster 1')

plt.scatter (x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter (x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label='Cluster 3')

plt.scatter (x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter (x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter (kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()


# In[ ]:





# In[ ]:




