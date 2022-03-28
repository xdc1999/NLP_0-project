# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 22:06:57 2022

@author: DC
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,HashingVectorizer
import pickle
import math
from nltk.corpus import stopwords
from nltk import everygrams
from nltk.stem import WordNetLemmatizer
import enchant

#name_list = ['n_2010','n_2011','n_2012']
path = r"C:\Users\DC\Desktop"
#for i in range(len(name_list))
with open(path+'\\'+ 'n_2019'+'.txt', 'rb') as file:
    b = pickle.load(file)
    
    #print(b)
    #print(type(b))
    #print(len(b))
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
                if not d.check(j): #append proper nouns even if they cannot be looked up in dic
                    lst1.append(j)
                else: #append lemmatization if it can be looked up in dic
                    lst1.append(wnl.lemmatize(j,pos='n'))
            elif d.check(j) and (j not in stop_words): #judge if it is a word
                j=j.lower() #lowering words helps de-duplication
                lst1.append(wnl.lemmatize(j,pos='n'))
    lst1=list(set(lst1))
    lst2.append(' '.join(lst1))
v = CountVectorizer(stop_words='english',max_df=0.98,min_df=0.02)#remove the top and bottom 10%
v.fit_transform(lst2)
word_list=v.get_feature_names()   
unique = word_list    
#data_dict = CountVectorizer(b,max_df = 0.75, min_df = 0.25)
d = list(b.values())
data_list = [i for k in d for i in k]
firmCnt = len(b)
wordCnt = len(data_list)

t = Thread(target=scrape, args=(5,))
t.start()

#TF-IDF
a = []
c = []
tf_idf = np.zeros((firmCnt,len(unique)))
for i in range(firmCnt):
    data = data_list[i]
    for j in range(len(unique)):
        word = unique[j]
        D_con = data_list.count(word)
        if word in data:
            fre = data.count(word)     
            tf = fre / len(unique)
            idf = math.log(len(data_list)/ (1 + D_con))
            tfidf = tf*idf
        else:
            tfidf = 0.0
        a.append(tfidf)
    c.append(a)
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
df_result.to_csv(path+'\\result_2019.csv')

             

from sklearn.cluster import KMeans
clf = KMeans(n_clusters=30) #参数4代表聚成4类，还有其他参数可以设置
s=clf.fit(tfidf) #调用fit()方法进行聚类 输入：weight=Tfidf的稀疏矩阵
print(s)          #可以查看kmeans()其他参数
 
#中心点
print(len(clf.cluster_centers_))

listlabel=[]
i=0
while i<len(clf.labels_):
    listlabel.append([i,clf.labels_[i]])
    i=i+1
 
frame = pd.DataFrame(listlabel,columns=['index','class'])#使用pandas处理方便
 


