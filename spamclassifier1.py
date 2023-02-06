# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 23:25:55 2023

@author: Abhyuday
"""
#cp1252

import pandas as pd

messages=pd.read_csv('spam.csv',sep=',',encoding='cp1252')
messages.columns
df=messages[['v1','v2']]
df.columns=['label','messages']
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
print(df.messages)

for i in range(0,len(df)):
    review=re.sub('[^a-zA-Z]', ' ', df['messages'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.20,random_state=0)


from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,Y_train)
y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_m=confusion_matrix(Y_test, y_pred)


from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test, y_pred)