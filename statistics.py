import pandas as pd
import nltk
import os
import regex as re
from nltk import *
import json
import string
from nltk.corpus import stopwords
import os

import numpy as np
import matplotlib as plt

data = pd.read_csv('./UoY.csv')

data['combined'] = data['Outcome'].astype(str) + ' ' + data['Objective'] + ' ' + data['Description']


department_list = data['Department'].unique()
department_avg = {}

for department in department_list:
    department_avg[department] = len(data[data['Department'] == department])

print(department_avg)
