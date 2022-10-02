import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import random
from itertools import repeat


def create_list_of_features(pos_texts, neg_texts, top_neg):
  pos_features = list(map(extract_features, pos_texts, repeat(top_neg)))
  pos_features = list(map(lambda x: (x, "pos"), pos_features))

  neg_features = list(map(extract_features, neg_texts, repeat(top_neg)))
  neg_features = list(map(lambda x: (x, "neg"), neg_features))

  return pos_features + neg_features


def extract_features(text, top_neg):
  features = dict()
  scores = SentimentIntensityAnalyzer().polarity_scores(text)
  wordcount = sum(
    list(
      map(lambda x: 1
          if x.lower() in top_neg else 0, nltk.word_tokenize(text))))

  features["compound_score"] = scores["compound"] + 1
  features["negative_score"] = scores["neg"]
  features["wordcount"] = wordcount

  return features


def train_model(features):
  # Use 1/4th for training
  count = len(features) // 4
  random.shuffle(features)
  classifier = nltk.NaiveBayesClassifier.train(features[:count])
  accuracy = nltk.classify.accuracy(classifier, features[count:])

  return classifier, accuracy