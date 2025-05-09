import csv

input_file = '/datasets/fake_data.csv'
output_file = '/datasets/fake_data_cleaned.csv'

with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) == 4:  # expecting 4 columns like title, text, subject, date
            writer.writerow(row)

input_file = '/datasets/true_data.csv'
output_file = '/datasets/true_data_cleaned.csv'

with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) == 4:  # expecting 4 columns like title, text, subject, date
            writer.writerow(row)

import pandas as pd
import string
import re
from sklearn.utils import shuffle

data_fake = pd.read_csv('/datasets/fake_data_cleaned.csv')
data_true = pd.read_csv('/datasets/true_data_cleaned.csv')
data_fake["class"] = 0
data_true["class"] = 1
data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge = data_merge.drop(["title", "subject", "date"], axis=1)
data = shuffle(data_merge)

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data['text'] = data['text'].apply(word_drop)
data.to_csv('/datasets/cleaned_news.csv', index=False)
