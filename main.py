import pandas as pd
from nltk import *
from keybert import KeyBERT
# nltk.download()
import string
from nltk.corpus import stopwords
import numpy as np
import matplotlib as plt

kw_model = KeyBERT()
data = pd.read_csv('./UoY.csv')

data['combined'] = data['Outcome'].astype(str) + ' ' + data['Objective'] + ' ' + data['Description']
data['preprocessed'] = ""
data['keywords'] = ""

# remove nan combined
data = data[data['combined'].notna()]

# stem and lemmatize
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
words_set = set()

for index, row in data.iterrows():
    combined = row['combined']
    combined = combined.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokenized_doc = word_tokenize(combined)
    tokenized_doc = [word for word in tokenized_doc if word not in stopwords.words('english')]
    lemmatized_list = []
    for token in tokenized_doc:
        # stemmed_token = stemmer.stem(token)
        lemmatized_token = lemmatizer.lemmatize(token)
        lemmatized_list.append(lemmatized_token)
        words_set.add(lemmatized_token)

    data.at[index, 'preprocessed'] = ' '.join(map(str, lemmatized_list))
    keywords = kw_model.extract_keywords(data.at[index, 'preprocessed'])
    data.at[index, 'keywords'] = keywords

data.to_csv('./tmp.csv')
newData = pd.read_csv('./tmp.csv')

# ini statistics

# crate new col in each row for store keywords

# extract keyword from (array of keys)->string

# compare similarity bwtween result of department kwywords
