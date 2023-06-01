#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from nltk.corpus import stopwords
from textblob import TextBlob


# In[77]:


df=pd.read_csv('Elon_musk.csv',encoding="latin-1")
df


# In[78]:


df['word_count'] = df['Text'].apply(lambda x: len(str(x).split(" ")))
df[['Text','word_count']].head(10)


# In[79]:


df['char_count'] = df['Text'].str.len()
df[['Text','char_count']].head(10)


# In[80]:


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))
df['avg_word'] = df['Text'].apply(lambda x: avg_word(x))
df[['Text','avg_word']].head(10)


# In[81]:


import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
df['stopwords'] = df['Text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['Text','stopwords']].head(10)


# In[82]:


df['hastags'] = df['Text'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
df[['Text','hastags']].head(10)


# In[83]:


df['numerics'] = df['Text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['Text','numerics']].head(10)


# In[84]:


df['upper'] = df['Text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['Text','upper']].head(10)


# In[85]:


df['Text'] = df['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['Text'].head(10)


# In[86]:


df['Text'] = df['Text'].str.replace('[^\w\s]','')
df['Text'].head(10)


# In[87]:


stop = stopwords.words('english')
df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['Text'].head()


# In[88]:


freq = pd.Series(' '.join(df['Text']).split()).value_counts()[:10]
freq


# In[89]:


freq = list(freq.index)
df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['Text'].head()


# In[90]:


freq = pd.Series(' '.join(df['Text']).split()).value_counts()[-10:]
freq


# In[91]:


freq = list(freq.index)
df['Text'] = df['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['Text'].head()


# In[92]:


df['Text'][:5].apply(lambda x: str(TextBlob(x).correct()))


# In[93]:


import nltk
nltk.download('punkt')
TextBlob(df['Text'][1]).words


# In[94]:


from nltk.stem import PorterStemmer
st = PorterStemmer()
df['Text'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


# In[95]:


from textblob import Word
import nltk
nltk.download('wordnet')
df['Text'] = df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['Text'].head()


# In[96]:


TextBlob(df['Text'][0]).ngrams(2)


# In[97]:


tf1 = (df['Text'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1


# In[98]:


for i,word in enumerate(tf1['words']):
 tf1.loc[i, 'idf'] = np.log(df.shape[0]/(len(df[df['Text'].str.contains(word)])))
tf1


# In[99]:


tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


# In[100]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
vect = tfidf.fit_transform(df['Text'])
vect


# # ###Bag of words

# In[101]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(df['Text'])
data_bow


# # ###Sentiment Analysis

# In[102]:


df['Text'][:5].apply(lambda x: TextBlob(x).sentiment)


# In[103]:


df['sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment[0] )
df[['Text','sentiment']].head(10)


# In[104]:


from wordcloud import WordCloud, STOPWORDS
def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off");stopwords = STOPWORDS


# In[105]:


df.plot.scatter(x='word_count',y='sentiment',figsize=(8,8),title='Sentence sentiment value to sentence word count')


# In[ ]:




