import nltk
from nltk.corpus import wordnet as wn
import csv
import numpy as np
from enum import Enum
UTTERANCES_NPY_FILENAME = "utterances.npy"
UTTERANCES_CSV_FILENAME = "utterances.csv"

class Activity(Enum):
	Bathing = 0
	Bed_Toilet_Transition = 1
	Eating = 2
	Enter_Home = 3
	Housekeeping = 4
	Leave_Home = 5
	Meal_Preparation = 6
	Personal_Hygiene = 7
	Sleep = 8
	Sleeping_Not_in_Bed = 9
	Wandering_in_room = 10
	Watch_TV = 11
	Work = 12

# TODO: is there some way to generalize this,
# in case other datasets have different activities?
# also, my root word choices are probably not the best
ACTIVITIES_TO_ROOT_WORDS = {
	Activity.Bathing:["bath"],
	# Activity.Bed_Toilet_Transition: ["bed"],
	Activity.Eating:["kitchen", "eat"],
	Activity.Enter_Home:["enter", "arrive"],
	Activity.Leave_Home:["leave"],
	Activity.Housekeeping:["housekeeping"],
	Activity.Meal_Preparation:["cook"],
	Activity.Personal_Hygiene:["bathroom"],
	Activity.Sleep:["sleep"],
	Activity.Sleeping_Not_in_Bed:["sleep"],
	Activity.Wandering_in_room:["wandering"],
	Activity.Watch_TV:["tv"],
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

	# utterances = np.load(UTTERANCES_NPY_FILENAME)
	# processed_utterances = [u + "#" for u in utterances]
	# for u in utterances:


	# print(utterances[0])
	# print(utterances.shape)
	# np.savetxt(open("processed_utterances.txt", "wb"), utterances, fmt="%s")
	# parse_utterances(utterances)

main()