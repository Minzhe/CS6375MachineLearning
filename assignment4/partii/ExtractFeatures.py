######################################################################
######                    Extract Features                       #####
######################################################################
# python ExtractFeatures.py --train data/20news-bydate-train/ --test data/20news-bydate-test/

import argparse
import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
import re
import json

parser = argparse.ArgumentParser(description='Extract features.')
parser.add_argument('--train', help='Path of train dataset')
parser.add_argument('--test', help='Path of test dataset')

args = parser.parse_args()
train_path = args.train
test_path = args.test

def getPathAndLabel(path):
    class_labels = os.listdir(path)
    path_label = []
    ### loop through every class folder
    for label in class_labels:
        class_folder = os.path.join(path, label)
        ### get into folder
        if os.path.isdir(class_folder):
            files = os.listdir(class_folder)
            for file in files:
                path_label.append((os.path.abspath(os.path.join(class_folder, file)), label))
    return path_label

def parseTextFile(path):
    text = open(path).read()
    ### get raw words
    raw = [w.lower() for w in re.findall(r'[a-zA-Z]+', text)]
    ### tokenize
    tokens = [w for w in raw if len(w) > 2 and w not in stopwords.words('english')]
    ### lemmatize
    wnl = nltk.WordNetLemmatizer()
    lemma = [wnl.lemmatize(t) for t in tokens]
    ### check if engish words
    en_words_set = set(words.words())
    words_list = [w for w in lemma if w in en_words_set]
    return words_list

def extractWords(path_label):
    features_data = dict()
    for (path, label) in path_label:
        name = os.path.basename(path)
        print('Parsing file {}'.format(name))
        try:
            features = parseTextFile(path=path)
            features_data[name] = (features, label)
        except UnicodeDecodeError:
            print('*** Error in reading file {}'.format(path))
    return features_data

train_path_label = getPathAndLabel(path=train_path)
test_path_label = getPathAndLabel(path=test_path)
train_data = extractWords(train_path_label)
test_data = extractWords(test_path_label)

with open('train_data_json.txt', 'w') as out_file:
    json.dump(train_data, out_file)
with open('test_data_json.txt', 'w') as out_file:
    json.dump(test_data, out_file)