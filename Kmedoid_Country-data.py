#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import seaborn as sns
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df=pd.read_csv('Country-data.csv')
print(df.head())


# In[41]:


df.isnull().sum()


# In[42]:


df.plot.scatter(x="income", y="inflation")
df.plot("income")
df.plot("inflation")


# In[43]:


sns.pairplot(df)
df


# In[44]:


df.drop(['country'],axis=1,inplace=True)
df


# In[45]:


scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)
scaled_df


# In[46]:


scaled_df=pd.DataFrame(scaled_df,columns=df.columns)

scaled_df.head()


# In[47]:


pca=PCA(n_components=2)
p_data=pca.fit_transform(scaled_df)


# In[62]:


km=KMedoids(n_clusters=3)
y_predict=km.fit_predict(p_data)
y_predict


# In[63]:


df3=pd.DataFrame(km.labels_).value_counts()
df3


# In[64]:


df['clusters']=y_predict
df.head()


# In[51]:


km.cluster_centers_


# In[65]:


scaled_df['clusters']=km.labels_
scaled_df


# In[53]:


from sklearn.metrics import silhouette_score


# In[54]:


score=silhouette_score(p_data,km.labels_ , metric='euclidean')
score


# In[55]:


df.to_csv('kmedoid_result.csv',index=False)


# In[56]:


sns.scatterplot(scaled_df['income'],scaled_df['gdpp'],hue='clusters',data=scaled_df) 
plt.title("income vs gdpp", fontsize=12)
plt.xlabel("income", fontsize=10)
plt.ylabel("gdpp", fontsize=10)
plt.show()


# In[57]:


df1=pd.read_csv('Country-data.csv')
df1['clusters']=y_predict
df1.tail()


# In[67]:


developed=df1[df1['clusters']==0]['country']
under_developed=df1[df1['clusters']==1]['country']
developing=df1[df1['clusters']==2]['country']

print("Number of developed countries",len(developed))
print("Number of under_developed countries",len(under_developed))
print("Number of developing countries",len(developing))


# In[68]:


print(developed)


# In[69]:


print(under_developed)


# In[61]:



print(developing)


# In[ ]:




