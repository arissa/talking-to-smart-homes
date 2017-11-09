import nltk
import csv
import numpy as np
UTTERANCES_NPY_FILENAME = "utterances.npy"
UTTERANCES_CSV_FILENAME = "utterances.csv"

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

# create a stateful vector that represents the lights that should be on or off during an activity.
# the stateful vector also has activity info.
def parse_utterances(utterances):
	# TODO
	pass

def main():
	# extract_utterances_from_csv(UTTERANCES_CSV_FILENAME, UTTERANCES_NPY_FILENAME)
	utterances = np.load(UTTERANCES_NPY_FILENAME)
	parse_utterances(utterances)
	
main()