#!/usr/bin/env python
# coding: utf-8

# # J002 ASSIGNMENT NLP

# ## SPAM DETECTION

# ### Importing Libraries

# In[1]:


# Basic Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Text preprocessing
import nltk as nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Model Building
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# ### Reading and viewing the data

# In[2]:


data = pd.read_csv('/Users/home/Downloads/spam.csv',encoding="ISO-8859-1")


# In[3]:


data


# ### 1. Exploratory Data Analysis and preprocessing :

# #### 1. Data shape and variable datatypes

# In[4]:


data.shape


# In[5]:


data.dtypes


# #### 2. Handling null values

# In[6]:


data.isnull().sum()


# ##### The columns Unnamed: 2, Unnamed: 3 and Unnamed: 4 have null values

# In[7]:


data.isnull().sum()*100/data.shape[0]


# ##### The columns Unnamed: 2, Unnamed: 3 and Unnamed: 4 have null values greater than 99% so we drop them

# In[8]:


data=data.drop(data.columns[[2,3,4]],axis=1)


# ##### The new data now looks like:

# In[9]:


data


# ##### Column v1 is the outcome and v2 is the predictor so we rename them to X and Y

# In[10]:


data.columns = ['Y' , 'X']


# #### 4. Checking the distribution of the target variable

# In[11]:


plt.figure(figsize=(5,5))
data['Y'].value_counts().plot(kind='bar',color='black',label='Spam Vs Not Spam')
plt.legend();


# #### 5. Preprocessing 1 - identifying stopwords, punctuations and creating a corpus

# In[12]:


trashitems = list(stopwords.words('english'))+list(punctuation)
stemmer = LancasterStemmer()
corpus = data['X'].tolist()


# In[13]:


corpus[0]


# In[14]:


len(corpus)


# #### 5. Preprocessing 2 - Modifying the corpus

# In[15]:


final_corpus = []
for i in range(len(corpus)):
    word = word_tokenize(corpus[i].lower())
    word = [stemmer.stem(y) for y in word if y not in trashitems]
    j = " ".join(word)
    final_corpus.append(j)


# #### 5. Preprocessing 3 - TF IDF Vectorization

# In[16]:


x=data['X']
y=data['Y']


# In[17]:


tfidf = TfidfVectorizer()
vector = tfidf.fit_transform(x)
x = vector.toarray()
x


# ### 2. Model building

# #### 1. Train test split (80:20)

# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# #### 2. Model building 1 - Multinomial Naive Bayes

# In[19]:


mnb = MultinomialNB()
mnb.fit(x_train,y_train)


# #### 2. Model prediction 1 - Multinomial Naive Bayes

# In[20]:


ypredmnb = mnb.predict(x_train)
ypredmnb


# #### 2. Model evaluation 1 - Multinomial Naive Bayes

# ##### 2.1. Accuracy score

# In[21]:


print(accuracy_score(y_train,ypredmnb)*100, '%')


# ##### 2.2. Confusion Matrix

# In[22]:


cnf_mnb = confusion_matrix(y_train,ypredmnb)
print(cnf_mnb)


# In[23]:


sns.heatmap(cnf_mnb, annot=True);


# ##### 2.3. Classification report

# In[24]:


print(classification_report(y_train,ypredmnb))


# #### 3. Model building 2 - Linear Support Vector Classifier

# In[25]:


svc = LinearSVC().fit(x_train,y_train)


# #### 3. Model prediction 2 - Linear Support Vector Classifier

# In[26]:


ypred_svc = svc.predict(x_train)


# #### 3. Model evaluation 2 - Linear Support Vector Classifier

# ##### 3.1. Accuracy score

# In[27]:


print(accuracy_score(y_train,ypred_svc)*100, '%')


# ##### 3.2. Confusion Matrix

# In[28]:


cnf_svc = confusion_matrix(y_train,ypred_svc)
print(cnf_svc)


# In[29]:


sns.heatmap(cnf_svc, annot=True);


# ##### 3.3. Classification report

# In[30]:


print(classification_report(y_train,ypred_svc))


# #### 4. Model building 3 - XG Boost

# In[31]:


clf = xgb.XGBClassifier()
clf.fit(x_train, y_train)  


# #### 4. Model prediction 3 - XG Boost

# In[32]:


ypred_xgb = clf.predict(x_train)
ypred_xgb


# #### 4. Model evaluation 3 - XG Boost

# ##### 4.1. Accuracy score

# In[33]:


print(accuracy_score(y_train,ypred_xgb))


# ##### 4.2. Confusion matrix

# In[34]:


cnf_xgb = confusion_matrix(y_train,ypred_xgb)
cnf_xgb


# In[35]:


sns.heatmap(cnf_xgb, annot=True);


# ##### 4.3. Classification report

# In[36]:


print(classification_report(y_train,ypred_xgb))


# In[ ]:




