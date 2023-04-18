#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Question 1: Reading the dataset in a dataframe
import pandas as pd

df=pd.read_csv(r"C:\Users\neel1\Desktop\Set_I.csv")
#passng na values
na1 = ["NA","?","Not Available","na","NA"]
df=pd.read_csv(r"C:\Users\neel1\Desktop\Set_I.csv",na_values=na1)


# In[2]:


#Question 2: Print Entire dataframe
print(df.to_string())


# In[3]:


df.isnull().sum()


# In[5]:


df['MINIMUM_PAYMENTS']=df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean())
df['PRC_FULL_PAYMENT']=df['PRC_FULL_PAYMENT'].fillna(df['PRC_FULL_PAYMENT'].mean())
df['TENURE']=df['TENURE'].fillna(df['TENURE'].mean())


# In[9]:


df.isnull().sum()
df.info()


# In[22]:


new_df =df.iloc[:, [1,2,3,4,5,6,7,8,9,10,11]]
print(new_df)


# In[24]:


from sklearn.cluster import KMeans
kmeans = KMeans(5)
kmeans.fit(new_df)


# In[25]:


pred = kmeans.predict(new_df)
pred


# In[26]:


frame = pd.DataFrame(new_df)
frame['cluster'] = pred
frame['cluster'].value_counts()


# In[30]:


from sklearn.preprocessing import OneHotEncoder

OneHotEncoder().fit_transform(new_df)

#you cannot do it directly if you have more than 3 columns. However, you can apply a Principal Component Analysis to reduce the space in 2 columns and visualize this instead.
from sklearn.decomposition import PCA

pca_num_components = 2

reduced_data = PCA(n_components=pca_num_components).fit_transform(new_df)
results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x="pca1", y="pca2", hue=frame['cluster'], data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()



# In[ ]:




