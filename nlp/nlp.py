import nltk
from nltk.corpus import wordnet as wn
import csv
import numpy as np
from enum import Enum
import glob, os
import pandas
import unidecode
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

# for BedToToilet amd SleepingInBed, how do we determine which bedroom is relevant?
# Otherwise the lights in all 3 bedrooms will turn on and off.
# This dictionary was created using the 2010 dataset.
# TODO: finish creating this dict, or figure out if there's a way to automate it
# activities_to_lights = {Activity.Bathing:[11, 16], Activity.BedToToilet: [7, 8, 9, 10, 16], Activity.Eating: [3, 4, 5]}


def extract_utterances_from_csv(list_of_csv_filenames, savetxt_filename):
	final_utterances = []
	final_labels = []
	for csv_filename in list_of_csv_filenames:
		print(csv_filename)
		with open(csv_filename, "rb") as f:
			reader = csv.DictReader(f)
			# print(reader.fieldnames)
			i = 0
			for row in reader:
				utterances = row['Answer.WritingTexts']
				labels = row['Answer.Activity']
				utterances = utterances.split("|")
				for u in utterances:
					# print(u)
					u = u.encode('utf-8')
					u = u.decode('ascii', 'ignore')
					u = str(u)
					# u = unidecode.unidecode(u)
					# u = u.encode('ascii', 'ignore')
				labels = labels.split("|")
				for label in labels:
					if label.lower() not in ACTIVITY_LABELS:
						print(i)
						print(label)
						print("LABEL IS NOT IN ACTIVITY LABELS. ERROR")
				final_utterances.extend(utterances)
				final_labels.extend(labels)
				i+=1
		if len(final_utterances) != len(final_labels):
				print("CSV IS NOT FORMATTED CORRECTLY. MISMATCHING NUMBER OF LABELS AND UTTERANCES.")

	final_utterances_and_labels = [None, None]
	final_utterances_and_labels[0] = final_utterances
	final_utterances_and_labels[1] = final_labels
	np.savetxt(open(savetxt_filename, "wb"), final_utterances_and_labels, fmt="%s", delimiter="|")

	print(str(len(final_utterances)) + " utterances and labels extracted and saved at " + savetxt_filename)


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
		
def from_labels_to_logical_representation():
	pass

def main():
	csv_filenames = []
	for file in glob.glob("utterances/*.csv"):
		csv_filenames.append(file)
	extract_utterances_from_csv(csv_filenames, "labeled_utterances.txt")


main()