#!/usr/bin/env python
# coding: utf-8

# In[100]:


#Question 1: Reading the dataset in a dataframe
import pandas as pd

df=pd.read_csv(r"C:\Users\neel1\Desktop\Set_II.csv")

#Question 2
#passng na values
na1 = ["NA","?","Not Available","na","NA"]
df=pd.read_csv(r"C:\Users\neel1\Desktop\Set_II.csv",na_values=na1)


# In[101]:


#Question 3: displaying entire dataset
print(df.to_string())


# In[102]:


#treating missing values

#age is replaced with mean as datatype is numeric
df['age'].fillna(df['age'].mean(),inplace=True)
#bp is replaced with median as datatype is numeric
df['bp'].fillna(df['bp'].median(),inplace=True)
#sg is replaced with mean as datatype is numeric
df['sg'].fillna(df['sg'].mean(),inplace=True)
#al is replaced with mean as datatype is numeric
df['al'].fillna(df['al'].mean(),inplace=True)
#su is replaced with median as datatype is numeric
df['su'].fillna(df['su'].median(),inplace=True)
#next all cols na is replaced by mode as datatype is object
df['rbc']=df['rbc'].fillna(df['rbc']).mode()[0]
df['pc']=df['pc'].fillna(df['pc']).mode()[0]
df['appet']=df['appet'].fillna(df['appet']).mode()[0]
df['pe']=df['pe'].fillna(df['pe']).mode()[0]
df['ane']=df['ane'].fillna(df['ane']).mode()[0]
df['classification'].fillna(method='bfill', inplace=True)


# In[103]:


#after treating missing values we are checking whether the missing values are removed or not
df.isnull().sum()
df.info()


# In[104]:


#printing unique and nunique values with its count
for col in df:
 print(col + " unique count = " + str(df[col].nunique()))
 print(df[col].unique(),end="\n\n")


# In[105]:


#Question 4: detecting outlier
#finding range fo specific col using IQR
Q1=df['bp'].quantile(0.25)
Q3=df['bp'].quantile(0.75)
IQR=Q3-Q1
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
print(upper)
print(lower)

#printing outlier
outliers=df['bp'][((df['bp']<(Q1-1.5*IQR)) |(df['bp']>(Q3+1.5*IQR)) )]


# In[106]:


#treating outlier on bp column
df['bp'].replace([95.0,55.0],[100.0,60.0],inplace=True)
#printing outlier if exists
outliers=df['bp'][((df['bp']<(Q1-1.5*IQR)) |(df['bp']>(Q3+1.5*IQR)) )]


# In[107]:


#same applying for rest columns which are numeric
Q1=df['sg'].quantile(0.25)
Q3=df['sg'].quantile(0.75)
IQR=Q3-Q1
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
print(upper)
print(lower)


#printing outlier
outliers=df['sg'][((df['sg']<(Q1-1.5*IQR)) |(df['sg']>(Q3+1.5*IQR)) )]


# In[108]:


#treating outlier on sg column
df['sg'].replace([1.0275000000000003,1.0074999999999998],[2,3],inplace=True)
#printing outlier if exists
outliers=df['sg'][((df['sg']<(Q1-1.5*IQR)) |(df['sg']>(Q3+1.5*IQR)) )]


# In[109]:


#same applying for rest columns which are numeric
Q1=df['al'].quantile(0.25)
Q3=df['al'].quantile(0.75)
IQR=Q3-Q1
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
print(upper)
print(lower)


#printing outlier
outliers=df['al'][((df['al']<(Q1-1.5*IQR)) |(df['al']>(Q3+1.5*IQR)) )]


# In[110]:


#treating outlier on al column
df['al'].replace([5.0,-3.0],[6.0,7.0],inplace=True)
#printing outlier if exists
outliers=df['al'][((df['al']<(Q1-1.5*IQR)) |(df['al']>(Q3+1.5*IQR)) )]


# In[111]:


#same applying for rest columns which are numeric
Q1=df['su'].quantile(0.25)
Q3=df['su'].quantile(0.75)
IQR=Q3-Q1
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
print(upper)
print(lower)


#printing outlier
outliers=df['su'][((df['su']<(Q1-1.5*IQR)) |(df['su']>(Q3+1.5*IQR)) )]


# In[112]:


#treating outlier on su column
df['su'].replace([0.0,0.0],[4,5],inplace=True)
#printing outlier if exists
outliers=df['su'][((df['su']<(Q1-1.5*IQR)) |(df['su']>(Q3+1.5*IQR)) )]


# In[113]:


df_encoded = pd.get_dummies(df, columns=['rbc','pc','appet','pe','ane'])

# print the encoded dataframe
print(df_encoded)


# In[114]:


df_encoded.info()


# In[116]:


X = df_encoded.values[:, 2:11]
Y = df_encoded.values[:,1]
print(X)
print(Y)


# In[117]:


#IMPORTING TRAIN_TEST_SPLIT FROM sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


# In[118]:


#IMPORTING decision tree classifier FROM sklearn
from sklearn.tree import DecisionTreeClassifier
#applying both gini and entropy method for checking accuracy
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[119]:


y_pred = clf_gini.predict(X_test)
y_pred


# In[120]:


from sklearn.metrics import accuracy_score
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)


# In[121]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[122]:


#importing classification report from sklearn
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

