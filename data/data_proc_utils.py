import pandas as pd
import string
import demoji
import re 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


def remove_emoji(text):
    dem = demoji.findall(text)
    for item in dem.keys(): 
        text = text.replace(item, "")
    return text

#removes usernames, numbers, punctuation, urls, emojis
def clean_text(tweets_list): 
    pattern = r'[0-9]'
    cl_text = []
    for textt in tweets_list:
        #print(textt)
        tt = re.sub(r'(\s)?@\w+', r'\1', textt)
        tt = re.sub('http://\S+|https://\S+', '', tt)
        t = tt.translate(str.maketrans('', '', string.punctuation))
        t = remove_emoji(t)
        t = re.sub(pattern, '', t)
        cl_text.append(t)
    return cl_text

def remove_stop_words(tweets_list,lang): 
    stop_words = set(stopwords.words(lang)) 
    list_tweets_without_stop_words = []
    for t in tweets_list: 
        t = str(t) #check tweet list is it string or not to comment this line
        tokens = word_tokenize(t)
        t_arr = []
        for w in tokens:
            if w not in stop_words:
                t_arr.append(w)
        list_tweets_without_stop_words.append(t_arr)
    return list_tweets_without_stop_words

def stemming(tweets_list_of_list, lang):
    stemmer = SnowballStemmer(lang)
    stemmed = [] 
    for tweet in tweets_list_of_list:
        s = []
        for wordd in tweet:
            s.append(stemmer.stem(wordd))
        stemmed.append(s)
    return stemmed

def remove_index_col(df):
    dff = df.pop(df.columns[0])
    return dff

def sample_df(df,pos_samples, neg_samples):
    pos = df[(df['label'] == 1)]
    neg = df[(df['label'] == 0)]
    pos = pos.sample(n=pos_samples)
    neg = neg.sample(n=neg_samples)
    return pd.concat([pos,neg]).sample(frac=1)
