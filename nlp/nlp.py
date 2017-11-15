import nltk
from nltk.corpus import wordnet as wn
import csv
import numpy as np
from enum import Enum
UTTERANCES_NPY_FILENAME = "utterances.npy"
UTTERANCES_CSV_FILENAME = "utterances.csv"

class Activity(Enum):
	Bathing = 1
	BedToToilet = 2
	Eating = 3
	EnteringHome = 4
	LeavingHome = 5
	Housekeeping = 6
	PreparingMeals = 7
	PersonalHygiene = 8
	SleepingInBed = 9
	SleepingNotInBed = 10
	WanderingInRoom = 11
	WatchTV = 12
	Work = 13

# TODO: is there some way to generalize this,
# in case other datasets have different activities?
# also, my root word choices are probably not the best
ACTIVITIES_TO_ROOT_WORDS = {
	Activity.Bathing:["bath"],
	# Activity.BedToToilet: ["bed"],
	Activity.Eating:["kitchen", "eat"],
	Activity.EnteringHome:["enter", "arrive"],
	Activity.LeavingHome:["leave"],
	Activity.Housekeeping:["housekeeping"],
	Activity.PreparingMeals:["cook"],
	Activity.PersonalHygiene:["bathroom"],
	Activity.SleepingInBed:["sleep"],
	Activity.SleepingNotInBed:["sleep"],
	Activity.WanderingInRoom:["wandering"],
	Activity.WatchTV:["tv"],
	Activity.Work:["work"]
	}

# for BedToToilet amd SleepingInBed, how do we determine which bedroom is relevant?
# Otherwise the lights in all 3 bedrooms will turn on and off.
# This dictionary was created using the 2010 dataset.
# TODO: finish creating this dict, or figure out if there's a way to automate it
# activities_to_lights = {Activity.Bathing:[11, 16], Activity.BedToToilet: [7, 8, 9, 10, 16], Activity.Eating: [3, 4, 5]}


def extract_utterances_from_csv(csv_filename, npy_filename):
	with open(csv_filename, "rb") as f:
		reader = csv.reader(f)
		data = [row for row in reader]
	utterances = []
	for i in range(1, len(data)):
		response = data[i][-1]
		response = response.split("|")
		utterances.extend(response)
	utterances = np.array(utterances)
	print(utterances)
	np.save(npy_filename, utterances)
	print(str(len(utterances)) + " utterances extracted and saved at " + npy_filename)


def get_activity(utterance, activity_to_synonym_dict):
	tokenized = nltk.word_tokenize(utterance)
	activity = None

	parts_of_speech = nltk.pos_tag(tokenized)
	is_noun_or_verb = lambda pos: (pos[:2] == 'NN' or pos[:2][0] == 'V')
	nouns_and_verbs =[word for (word, pos) in parts_of_speech if is_noun_or_verb(pos)]
	for word in nouns_and_verbs:
		for a in activity_to_synonym_dict:
			if word in activity_to_synonym_dict[a]:
				activity = a
				break
	if not activity:
		print("This utterance does not have a corresponding activity.")
	return activity


def get_relevant_lights(activity):
	pass

# returns a Set of synonyms for a given word
def get_synonyms(word):
	synonyms = set()
	for ss in wn.synsets(word):
		lemmas = ss.lemma_names()
		for lemma in lemmas:
			synonyms.add(lemma.lower())
	return synonyms

def create_activity_to_synonym_dict():
	activity_to_synonym_dict = {}
	for activity in ACTIVITIES_TO_ROOT_WORDS:
		synonyms = set()
		for root_word in ACTIVITIES_TO_ROOT_WORDS[activity]:
			synonyms.update(get_synonyms(root_word))
		activity_to_synonym_dict[activity] = synonyms
	return activity_to_synonym_dict

# create a vector that represents which lights should be on or off, and contains activity data.
# this vector will be used to craft the loss function.
def parse_utterances(utterances):
	activity_to_synonym_dict = create_activity_to_synonym_dict()

	on = False
	on_phrases = ["turn on", "lights on"]
	off_phrases = ["turn off", "lights off"]
	vectors = []
	for utterance in utterances:
		utterance = utterance.lower()
		for phrase in on_phrases:
			if phrase in utterance:
				on = True
		for phrase in off_phrases:
			if phrase in utterance:
				on = False
		activity = get_activity(utterance, activity_to_synonym_dict)
		print(utterance, activity)
		# TODO get relevant lights based on activity
		# also, how do we handle unidentifiable activities?
		

def main():
	utterances = np.load(UTTERANCES_NPY_FILENAME)
	parse_utterances(utterances)

main()