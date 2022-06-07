import pandas as pd
from nltk import *
from keybert import KeyBERT
# nltk.download()
import string
from nltk.corpus import stopwords
import numpy as np
import matplotlib as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules


# kw_model = KeyBERT()
data = pd.read_csv('./UoY.csv')
data.fillna('')
data['combined'] = data['Outcome'].astype(str) + ' ' + data['Objective'] + ' ' + data['Description']
# data['preprocessed'] = ""
# data['keywords'] = ""

# # remove nan combined
# data = data[data['combined'].notna()]

# # stem and lemmatize
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
# words_set = set()

# for index, row in data.iterrows():
#     print(index)
#     combined = row['combined']
#     combined = combined.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
#     tokenized_doc = word_tokenize(combined)
#     tokenized_doc = [word for word in tokenized_doc if word not in stopwords.words('english')]
#     lemmatized_list = []
#     for token in tokenized_doc:
#         # stemmed_token = stemmer.stem(token)
#         lemmatized_token = lemmatizer.lemmatize(token)
#         lemmatized_list.append(lemmatized_token)
#         words_set.add(lemmatized_token)

#     data.at[index, 'preprocessed'] = ' '.join(map(str, lemmatized_list))
    # keywords = kw_model.extract_keywords(data.at[index, 'preprocessed'])
    # data.at[index, 'keywords'] = keywords

# data.to_csv('./tmp2.csv')
# newData = pd.read_csv('./tmp2.csv')



# ini statistics

# crate new col in each row for store keywords

# extract keyword from (array of keys)->string

# compare similarity bwtween result of department kwywords

# df = list()

# for index, row in newData.iterrows():
    
#     list = eval(row['keywords'])
#     list = [x[0] for x in list]
#     # list = list[0]
#     df.append(list)
#     # print(list)

# te = TransactionEncoder()
# te_ary = te.fit(df).transform(df)
# df = pd.DataFrame(te_ary, columns=te.columns_)
# df.to_csv('./tmp4.csv', index=False)

df = pd.read_csv('./tmp4.csv', index_col=False)

x = apriori(df, min_support=0.01, use_colnames=True)
y = association_rules(x, min_threshold=0)

print(y)
y.to_csv('./tmp3.csv')
