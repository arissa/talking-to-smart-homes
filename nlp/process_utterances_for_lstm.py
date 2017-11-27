import os
import time
import string
import pickle
import csv
import numpy as np
import sys
from operator import itemgetter
from collections import Counter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

# ACTIVITIES = {
#   Bathing: "Bathing",
#   Bed_Toilet_Transition: "Bed_Toilet_Transition",
#   Eating: "Eating",
#   Enter_Home: "Enter_Home",
#   Housekeeping: "Housekeeping",
#   Leave_Home: "Leave_Home",
#   Meal_Preparation: "Meal_Preparation",
#   Personal_Hygiene: "Personal_Hygiene",
#   Sleep: "Sleep",
#   Sleeping_Not_in_Bed: "Sleeping_Not_in_Bed",
#   Wandering_in_room: "Wandering_in_room",
#   Watch_TV: "Watch_TV",
#   Work: "Work",
# }

# ACTIVITY_LABELS = {
#   "Bathing": 0,
#   "Bed_Toilet_Transition": 1,
#   "Eating": 2,
#   "Enter_Home": 3,
#   "Housekeeping": 4,
#   "Leave_Home": 5,
#   "Meal_Preparation": 6,
#   "Personal_Hygiene": 7,
#   "Sleep": 8,
#   "Sleeping_Not_in_bed": 9,
#   "Wandering_in_room": 10,
#   "Watch_TV": 11,
#   "Work": 12,
# }

NUM_ACTIVITY_LABELS = 13

ACTIVITY_LABELS = {
    "bathing": 0,
    "bed_toilet_transition": 1,
    "eating": 2,
    "enter_home": 3,
    "housekeeping": 4,
    "leave_home": 5,
    "meal_preparation": 6,
    "personal_hygiene": 7,
    "sleep": 8,
    "sleeping_not_in_bed": 9,
    "wandering_in_room": 10,
    "watch_tv": 11,
    "work": 12,
}


def process_utterance(utterance):
    lemmatizer = WordNetLemmatizer()

    punct = set(string.punctuation)
    stopwords = set(sw.words('english'))
    lemmas = set()
    # Break the utterance into part of speech tagged tokens
    for token, tag in pos_tag(wordpunct_tokenize(utterance)):
        # Apply preprocessing to the token
        token = token.lower()
        token = token.strip()
        token = token.strip('_')
        token = token.strip('*')

        # If punctuation or stopword, ignore token and continue
        if token in stopwords or all(char in punct for char in token):
            continue

        # Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        # tag to perform much more accurate WordNet lemmatization.
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)
        lemma = lemmatizer.lemmatize(token, tag)
        lemmas.add(lemma)
    return lemmas


def create_lexicon_and_word_features(utterances):
    word_features = [None] * len(utterances)

    lexicon = set()
    for i in range(len(utterances)):
        lemmas = process_utterance(utterances[i])
        word_features[i] = list(lemmas)
        # print(lemma)
        lexicon.update(lemmas)

    lexicon = list(lexicon)
    return lexicon, word_features
    # print(lexicon)
    # print(word_features)


def create_features(lexicon, word_features):

    lemma_to_index_dict = {}
    for index, lemma in enumerate(lexicon):
        if lemma not in lemma_to_index_dict:
            lemma_to_index_dict[lemma] = index

    # total features matrix will be num_samples x num_words_in_lexicon
    total_features = np.zeros((len(word_features), 1,  len(lexicon)))
    for i in range(len(word_features)):
        features = np.zeros((1, len(lexicon)))
        words = word_features[i]
        for word in words:
            index = lemma_to_index_dict[word]
            features[:, index] = 1
        total_features[i] = features

    return total_features


def create_labels(labels_string):
    total_labels = np.zeros((len(labels_string),1, NUM_ACTIVITY_LABELS))
    for i in range(len(labels_string)):
        one_hot_label = np.zeros((1, NUM_ACTIVITY_LABELS))
        label = labels_string[i].lower().strip()
        index = ACTIVITY_LABELS[label]
        one_hot_label[:,index] = 1
        total_labels[i] = one_hot_label
    return total_labels

if __name__ == "__main__":

    X = []
    y = []
    with open("labeled_utterances.txt", "r") as f:
        lines = f.readlines()
    utterances = lines[0].split("|")
    for utterance in utterances:
        # print(utterance)
        utterance = utterance.decode('utf-8')
    X = utterances
    y = lines[1].split("|")
    lexicon, word_features = create_lexicon_and_word_features(utterances)
    features = create_features(lexicon, word_features)
    labels = create_labels(y)
    features.dump("data/features.npy")
    labels.dump("data/labels.npy")
    print(features)
    print(labels)
