#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


new_df = df.sample(30000,random_state=2) # random state 2 so we get the same set of questions every time we run it.


# In[6]:


new_df.isnull().sum()


# In[7]:


new_df.head()


# In[8]:


new_df.isnull().sum()


# In[9]:


new_df.duplicated().sum()


# In[10]:


# Distribution of duplicate and non-duplicate questions

print(new_df['is_duplicate'].value_counts())
print((new_df['is_duplicate'].value_counts()/new_df['is_duplicate'].count())*100)
new_df['is_duplicate'].value_counts().plot(kind='bar')


# In[11]:


# Repeated questions

qid = pd.Series(new_df['qid1'].tolist() + new_df['qid2'].tolist())
print('Number of unique questions',np.unique(qid).shape[0])
x = qid.value_counts()>1
print('Number of questions getting repeated',x[x].shape[0])


# In[12]:


# Repeated questions histogram

plt.hist(qid.value_counts().values,bins=160)
plt.yscale('log')
plt.show()


# In[13]:


# Feature Engineering

new_df['q1_len'] = new_df['question1'].str.len() 
new_df['q2_len'] = new_df['question2'].str.len()


# In[14]:


new_df.head()


# In[15]:


new_df['q1_num_words'] = new_df['question1'].apply(lambda row: len(row.split(" ")))
new_df['q2_num_words'] = new_df['question2'].apply(lambda row: len(row.split(" ")))
new_df.head()


# In[16]:


def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return len(w1 & w2)


# In[17]:


new_df['word_common'] = new_df.apply(common_words, axis=1)
new_df.head()


# In[18]:


def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return (len(w1) + len(w2))


# In[19]:


new_df['word_total'] = new_df.apply(total_words, axis=1)
new_df.head()


# In[20]:


new_df['word_share'] = round(new_df['word_common']/new_df['word_total'],2)
new_df.head()


# In[21]:


# Analysis of features
sns.displot(new_df['q1_len'])
print('minimum characters',new_df['q1_len'].min())
print('maximum characters',new_df['q1_len'].max())
print('average num of characters',int(new_df['q1_len'].mean()))


# In[22]:


sns.displot(new_df['q2_len'])
print('minimum characters',new_df['q2_len'].min())
print('maximum characters',new_df['q2_len'].max())
print('average num of characters',int(new_df['q2_len'].mean()))


# In[23]:


sns.displot(new_df['q1_num_words'])
print('minimum words',new_df['q1_num_words'].min())
print('maximum words',new_df['q1_num_words'].max())
print('average num of words',int(new_df['q1_num_words'].mean()))


# In[24]:


sns.displot(new_df['q2_num_words'])
print('minimum words',new_df['q2_num_words'].min())
print('maximum words',new_df['q2_num_words'].max())
print('average num of words',int(new_df['q2_num_words'].mean()))


# In[25]:


# common words
sns.distplot(new_df[new_df['is_duplicate'] == 0]['word_common'],label='non duplicate')
sns.distplot(new_df[new_df['is_duplicate'] == 1]['word_common'],label='duplicate')
plt.legend()
plt.show()


# In[26]:


# total words
sns.distplot(new_df[new_df['is_duplicate'] == 0]['word_total'],label='non duplicate')
sns.distplot(new_df[new_df['is_duplicate'] == 1]['word_total'],label='duplicate')
plt.legend()
plt.show()


# In[27]:


# word share
sns.distplot(new_df[new_df['is_duplicate'] == 0]['word_share'],label='non duplicate')
sns.distplot(new_df[new_df['is_duplicate'] == 1]['word_share'],label='duplicate')
plt.legend()
plt.show()


# In[28]:


ques_df = new_df[['question1','question2']]
ques_df.head()


# In[29]:


final_df = new_df.drop(columns=['id','qid1','qid2','question1','question2'])
print(final_df.shape)
final_df.head()


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
# merge texts
questions = list(ques_df['question1']) + list(ques_df['question2'])

cv = CountVectorizer(max_features=3000)
q1_arr, q2_arr = np.vsplit(cv.fit_transform(questions).toarray(),2)


# In[31]:


temp_df1 = pd.DataFrame(q1_arr, index= ques_df.index)
temp_df2 = pd.DataFrame(q2_arr, index= ques_df.index)
temp_df = pd.concat([temp_df1, temp_df2], axis=1)
temp_df.shape


# In[32]:


final_df = pd.concat([final_df, temp_df], axis=1)
print(final_df.shape)
final_df.head()


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2,random_state=1)


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[35]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
accuracy_score(y_test,y_pred)


# In[1]:


# Advanced Features
# 1. Token Features
# cwc_min: This is the ratio of the number of common words to the length of the smaller question
# cwc_max: This is the ratio of the number of common words to the length of the larger question
# csc_min: This is the ratio of the number of common stop words to the smaller stop word count among the two questions
# csc_max: This is the ratio of the number of common stop words to the larger stop word count among the two questions
# ctc_min: This is the ratio of the number of common tokens to the smaller token count among the two questions
# ctc_max: This is the ratio of the number of common tokens to the larger token count among the two questions
# last_word_eq: 1 if the last word in the two questions is same, 0 otherwise
# first_word_eq: 1 if the first word in the two questions is same, 0 otherwise
# 2. Length Based Features
# mean_len: Mean of the length of the two questions (number of words)
# abs_len_diff: Absolute difference between the length of the two questions (number of words)
# longest_substr_ratio: Ratio of the length of the longest substring among the two questions to the length of the smaller question
# 3. Fuzzy Features
# fuzz_ratio: fuzz_ratio score from fuzzywuzzy
# fuzz_partial_ratio: fuzz_partial_ratio from fuzzywuzzy
# token_sort_ratio: token_sort_ratio from fuzzywuzzy
# token_set_ratio: token_set_ratio from fuzzywuzzy


# In[ ]:




