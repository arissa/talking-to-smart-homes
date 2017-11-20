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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts



def identity(arg):
	"""
	Simple identity function works as a passthrough.
	"""
	return arg


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
	"""
	Transforms input data by using NLTK tokenization, lemmatization, and
	other normalization and filtering techniques.
	"""

	def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
		"""
		Instantiates the preprocessor, which make load corpora, models, or do
		other time-intenstive NLTK data loading.
		"""
		self.lower      = lower
		self.strip      = strip
		self.stopwords  = set(stopwords) if stopwords else set(sw.words('english'))
		self.punct      = set(punct) if punct else set(string.punctuation)
		self.lemmatizer = WordNetLemmatizer()

	def fit(self, X, y=None):
		"""
		Fit simply returns self, no other information is needed.
		"""
		return self

	def inverse_transform(self, X):
		"""
		No inverse transformation
		"""
		return X

	def transform(self, X):
		"""
		Actually runs the preprocessing on each document.
		"""
		return [
			list(self.tokenize(doc)) for doc in X
		]

	def tokenize(self, document):
		"""
		Returns a normalized, lemmatized list of tokens from a document by
		applying segmentation (breaking into sentences), then word/punctuation
		tokenization, and finally part of speech tagging. It uses the part of
		speech tags to look up the lemma in WordNet, and returns the lowercase
		version of all the words, removing stopwords and punctuation.
		"""
		# Break the document into sentences
		for sent in sent_tokenize(document):
			# Break the sentence into part of speech tagged tokens
			for token, tag in pos_tag(wordpunct_tokenize(sent)):
				# Apply preprocessing to the token
				token = token.lower() if self.lower else token
				token = token.strip() if self.strip else token
				token = token.strip('_') if self.strip else token
				token = token.strip('*') if self.strip else token

				# If punctuation or stopword, ignore token and continue
				if token in self.stopwords or all(char in self.punct for char in token):
					continue

				# Lemmatize the token and yield
				lemma = self.lemmatize(token, tag)
				yield lemma

	def lemmatize(self, token, tag):
		"""
		Converts the Penn Treebank tag to a WordNet POS tag, then uses that
		tag to perform much more accurate WordNet lemmatization.
		"""
		tag = {
			'N': wn.NOUN,
			'V': wn.VERB,
			'R': wn.ADV,
			'J': wn.ADJ
		}.get(tag[0], wn.NOUN)

		return self.lemmatizer.lemmatize(token, tag)



def build_and_evaluate(X, y, classifier=SGDClassifier, outpath=None, verbose=True):
	"""
	Builds a classifer for the given list of documents and targets in two
	stages: the first does a train/test split and prints a classifier report,
	the second rebuilds the model on the entire corpus and returns it for
	operationalization.

	X: a list or iterable of raw strings, each representing a document.
	y: a list or iterable of labels, which will be label encoded.

	Can specify the classifier to build with: if a class is specified then
	this will build the model with the Scikit-Learn defaults, if an instance
	is given, then it will be used directly in the build pipeline.

	If outpath is given, this function will write the model as a pickle.
	If verbose, this function will print out information to the command line.
	"""

	def build(classifier, X, y=None):
		"""
		Inner build function that builds a single model.
		"""
		if isinstance(classifier, type):
			classifier = classifier()

		model = Pipeline([
			('preprocessor', NLTKPreprocessor()),
			('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
			('classifier', classifier),
		])

		model.fit(X, y)
		return model

	# Label encode the targets
	labels = LabelEncoder()
	y = labels.fit_transform(y)

	# Begin evaluation
	if verbose: print("Building for evaluation")
	X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
	model = build(classifier, X_train, y_train)

	if verbose: print("Classification Report:\n")

	y_pred = model.predict(X_test)
	print(clsr(y_test, y_pred, target_names=labels.classes_))

	if verbose: print("Building complete model and saving ...")
	model = build(classifier, X, y)
	model.labels_ = labels


	if outpath:
		with open(outpath, 'wb') as f:
			pickle.dump(model, f)

		print("Model written out to {}".format(outpath))

	return model



if __name__ == "__main__":
	PATH = "model.pickle"
	if not os.path.exists(PATH):
		print(sys.getdefaultencoding())
		# Time to build the model
		from nltk.corpus import movie_reviews as reviews
		UTTERANCES_NPY_FILENAME = "utterances.npy"

		utterances = np.load(UTTERANCES_NPY_FILENAME)
		X = [u.decode('utf-8') for u in utterances]
		y = [foo for foo in np.random.randint(1, 10, len(X))]
		print(X)
		# print(len(y))
		model = build_and_evaluate(X,y, outpath=PATH)

	else:
		with open(PATH, 'rb') as f:
			model = pickle.load(f)

