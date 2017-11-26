import os
import time
import string
import pickle
import csv
import numpy as np
import sys
from operator import itemgetter

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

def create_lexicon(utterances):
	lemmatizer = WordNetLemmatizer()

	punct = set(string.punctuation)
	stopwords = set(sw.words('english'))

	lexicon = set()
	for utterance in utterances:
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
			# print(lemma)
			lexicon.add(lemma)

	return lexicon




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
	print(create_lexicon(utterances))


