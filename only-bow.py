#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


new_df = df.sample(30000)


# In[6]:


new_df.isnull().sum()


# In[7]:


new_df.duplicated().sum()


# In[8]:


ques_df = new_df[['question1','question2']] # new Dataframe where only question 1 and 2 are there.
ques_df.head()


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])  #question 1 and 2 inserted in a list.

cv = CountVectorizer(max_features=3000)  #make features of 3000 most used words.
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)  #split 60k questions into 30k each in q1 and q2_arr.


# In[10]:


temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)  #convert q1 and q2 into Dataframe and concatenate them.
temp_df.shape


# In[11]:


temp_df


# In[12]:


temp_df['is_duplicate'] = new_df['is_duplicate']


# In[13]:


temp_df.head()


# In[14]:


from sklearn.model_selection import train_test_split #data is ready we did train test split.
X_train,X_test,y_train,y_test = train_test_split(temp_df.iloc[:,0:-1].values,temp_df.iloc[:,-1].values,test_size=0.2,random_state=1)


# In[16]:


from sklearn.ensemble import RandomForestClassifier  #random forest used got 74% accuracy.
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[18]:


from xgboost import XGBClassifier  #used XG boost also.
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
accuracy_score(y_test,y_pred)


# In[ ]:




