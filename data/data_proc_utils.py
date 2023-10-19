import pandas as pd
import string
import demoji
import re 

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

def remove_index_col(df):
    dff = df.pop(df.columns[0])
    return dff

def sample_df(df,pos_samples, neg_samples):
    pos = df[(df['label'] == 1)]
    neg = df[(df['label'] == 0)]
    pos = pos.sample(n=pos_samples)
    neg = neg.sample(n=neg_samples)
    return pd.concat([pos,neg]).sample(frac=1)

