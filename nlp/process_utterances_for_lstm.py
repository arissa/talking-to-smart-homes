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
	word_features = [None]*len(utterances)

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
	total_features = np.zeros((len(word_features), len(lexicon)))
	for i in range(len(word_features)):
		features = np.zeros(len(lexicon))
		words = word_features[i]
		for word in words:
			index = lemma_to_index_dict[word]
			features[index] = 1
		total_features[i] = features

	print(total_features)
	print(np.sum(total_features))
	return total_features
# def trim_lexicon(lexicon):
#   w_counts = Counter(lexicon)
#   print(w_counts)

# def create_features(lexicon, utterances):


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
	create_features(lexicon, word_features)

