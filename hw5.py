import spacy
from newsapi import NewsApiClient
import pickle
import string
import pandas as pd
import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt
from wordcloud import WordCloud 
from textblob import TextBlob

def get_keywords_eng(text):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    for token in doc:
        if token.text in nlp_eng.Defaults.stop_words or token.text in string.punctuation:
            continue
        elif token.pos_ in pos_tag:
            result.append(token.text)
    return result

if __name__ == '__main__':
    data = []
    results = []

    nlp_eng = spacy.load('en_core_web_lg')
    newsapi = NewsApiClient(api_key='432bd495bfd049e596588ab9f6557278')

    articles = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-09-28', to='2021-10-28', sort_by='relevancy')

    filename = 'articlesCOVID.pckl'
    pickle.dump(articles, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))

    filepath = '/Users/andya/OneDrive/Desktop/Fall 2021/CS 4650 Big Data Analytics and Cloud Computing/articlesCOVID.pckl'
    pickle.dump(loaded_model, open(filepath, 'wb'))

    for i, article in enumerate(articles):
        for x in articles['articles']:
            title = x['title']
            date = x['publishedAt']
            description = x['description']
            content = x['content']
            data.append({'title': title, 'date': date, 'desc': description, 'content': content})

    df = pd.DataFrame(data)
    df = df.dropna()
    print(df.head())

    for content in df.content.values:
        results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])

    df['keywords'] = results

    df.to_csv('output.csv', index=False)

    text = str(results)
    wordcloud = WordCloud(width=900, height=500,max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()