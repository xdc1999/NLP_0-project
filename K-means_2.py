# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:05:36 2022

@author: DC
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,HashingVectorizer
import pickle
from nltk.corpus import stopwords
from nltk import everygrams
from nltk.stem import WordNetLemmatizer
import enchant
from sklearn.cluster import KMeans

#name_list = ['n_2010','n_2011','n_2012']
path = r"C:\Users\DC\Desktop"
#for i in range(len(name_list))
with open(path+'\\'+ 'n_2018'+'.txt', 'rb') as file:
    b = pickle.load(file)
    
    #print(b)
    #print(type(b))
    #print(len(b))
corpus = list(b)
stop_words=stopwords.words('english')
stop_words.extend(['item','general','business','overview'])
#from nltk.stem import PorterStemmer
#ps = PorterStemmer()

wnl = WordNetLemmatizer()
#wnl.lemmatize('dogs')

d = enchant.Dict("en_US")
#d.check("Hello")

lst1=[]
lst2=[]

cik = b.keys()
for i in cik:
    w=b[i]
    lst1=[]
    for j in w:
        if j.isalpha() and len(j)>1:

            if j.isupper(): #judge if it is a proper noun
                j=j.lower()
                #append proper nouns even if they cannot be looked up in dic
                if not d.check(j): 
                    lst1.append(j)
                else: #append lemmatization if it can be looked up in dic
                    lst1.append(wnl.lemmatize(j,pos='n'))
            elif d.check(j) and (j not in stop_words): #judge if it is a word
                j=j.lower() #lowering words helps de-duplication
                lst1.append(wnl.lemmatize(j,pos='n'))
    lst1=list(set(lst1))
    lst2.append(' '.join(lst1))
#remove the top and bottom 10%
v = CountVectorizer(stop_words='english',max_df=0.98,min_df=0.02)
v = v.fit_transform(lst2)





word_list=v.get_feature_names()   
unique = word_list    
#data_dict = CountVectorizer(b,max_df = 0.75, min_df = 0.25)




#Convert words in the text into a word frequency matrix 
vectorizer=CountVectorizer()
#Count the TF-IDF of each word
transformer=TfidfTransformer()
#The first fit_transform is to calculate the TF-IDF
#the second is to convert words in the text into a word frequency matrix
tfidf=transformer.fit_transform(vectorizer.fit_transform(lst2))
weight=tfidf.toarray()

#Divide firms into 30 classes
num_clusters = 30
km_cluster = KMeans(n_clusters=num_clusters,
                    init='k-means++')
result = km_cluster.fit_predict(weight)





cik = list(cik)
firm = {}
for i in range(30):
    a = []
    for j in range(len(result)):
        if result[j] == i:
            a.append(cik[j])
    firm[str(i)] = a
df1 = pd.Series(firm,index = firm.keys())
num = []
for i in range(30):
    a = []
    for j in range(len(result)):
        if result[j] == i:
            a.append(list(cik[j]))
    num.append(len(a))
df2 = pd.Series(num,index = firm.keys())
df_result = pd.concat([df1,df2],axis=1)
df_result.to_csv(path+'\\result_2018.csv')