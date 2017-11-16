######################################################################
######                   NaiveBayesClassifier                    #####
######################################################################

import json
import math
import argparse
from collections import defaultdict
from nltk.probability import FreqDist, DictionaryProbDist, LaplaceProbDist

parser = argparse.ArgumentParser(description='Naive Bayes Classifer.')
parser.add_argument('--train', help='Path of train dataset')
parser.add_argument('--test', help='Path of test dataset')

args = parser.parse_args()
train_path = args.train
test_path = args.test

class NaiveBayesClassifier(object):
    def __init__(self, label_pdist, feature_pdist):
        self._label_pdist = label_pdist
        self._feature_pdist = feature_pdist
        self._labels = list(label_pdist.samples())

    def labels(self):
        return self._labels

    def classify(self, featureset):
        return self.prob_classify(featureset).max()

    def prob_classify(self, featureset):
        featureset = featureset.copy()
        for feature_name in list(featureset.keys()):
            for label in self._labels:
                if (label, feature_name) in self._feature_pdist:
                    break
            else:
                del featureset[feature_name]

        # Find the log probabilty of each label, given the features.
        # Start with the log probability of the label itself.
        logprob = {}
        for label in self._labels:
            logprob[label] = self._label_pdist.logprob(label)

        # Then add in the log probability of features given labels.
        for label in self._labels:
            for (feature_name, feature_val) in featureset.items():
                if (label, feature_name) in self._feature_pdist:
                    feature_probs = self._feature_pdist[label, feature_name]
                    logprob[label] += feature_probs.logprob(feature_val)
                else:
                    logprob[label] += math.log(1e-30, 2) # = -INF.

        return DictionaryProbDist(logprob, normalize=True, log=True)

    @classmethod
    def train(cls, features_labels, LaplaceEstimator=LaplaceProbDist):
        """
        :param labeled_featuresets: A list of classified featuresets (featureset, label)
        """
        label_fdist = FreqDist()
        feature_fdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        feature_names = set()

        # Count up how many times each feature value occurred, given the label and featurename.
        for features, label in features_labels:
            label_fdist[label] += 1
            for feature_name, feature_val in features.items():
                feature_fdist[label, feature_name][feature_val] += 1
                feature_values[feature_name].add(feature_val)
                feature_names.add(feature_name)

        for label in label_fdist:
            n_samples = label_fdist[label]
            for feature_name in feature_names:
                count = feature_fdist[label, feature_name].N()
                if n_samples - count > 0:
                    feature_fdist[label, feature_name][None] += n_samples - count
                    feature_values[feature_name].add(None)

        # Create the P(label) distribution
        label_probdist = LaplaceEstimator(label_fdist)

        # Create the P(fval|label, fname) distribution
        feature_probdist = {}
        for ((label, fname), freqdist) in feature_fdist.items():
            probdist = LaplaceEstimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[label, fname] = probdist

        return cls(label_probdist, feature_probdist)


def wordFeatures(text, word_features):
    '''
    Function to count the feature words
    '''
    word_count_features = dict.fromkeys(word_features, 0)
    for word in text:
        if word in word_features:
            word_count_features[word] += 1
    return word_count_features

with open(train_path) as train_file:
    train_data_list = json.load(train_file)
with open(test_path) as test_file:
    test_data_list = json.load(test_file)

### get the most common 3000 words
train_data = list()
for entry in train_data_list:
    train_data.extend(entry[0])
fdist = FreqDist(train_data)
word_features = [word[0] for word in fdist.most_common(3000)]

### train data
train_data = list()
for entry in train_data_list:
    ### features and label
    train_data.append((wordFeatures(entry[0], word_features), entry[1]))

### test data
test_data = list()
for entry in test_data_list:
    ### features and label
    test_data.append((wordFeatures(entry[0], word_features), entry[1]))

nbclassifer = NaiveBayesClassifier.train(train_data)
correct_pred = 0
for test_feature, label in test_data:
    pred_label = nbclassifer.classify(test_feature)
    if pred_label == label:
        correct_pred += 1
accuracy_test = correct_pred / len(test_data) * 100

correct_pred = 0
for test_feature, label in train_data:
    pred_label = nbclassifer.classify(test_feature)
    if pred_label == label:
        correct_pred += 1
accuracy_train = correct_pred / len(train_data) * 100

print('Predction accuracy in train dataset is {}%'.format(round(accuracy_train, 5)))
print('Predction accuracy in test dataset is {}%'.format(round(accuracy_test, 5)))
