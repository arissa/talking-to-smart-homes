import numpy as np

filename = "twor.2010/data"
num_activities = 13
num_sensors = 51
window_size = 20
activity_encoder_map = {
	"Bathing" : 0,
	"Bed_Toilet_Transition" : 1,
	"Eating" : 2,
	"Enter_Home" : 3,
	"Housekeeping" : 4,
	"Leave_Home" : 5,
	"Meal_Preparation" : 6,
	"Personal_Hygiene" : 7,
	"Sleep" : 8,
	"Sleeping_Not_in_Bed" : 9,
	"Wandering_in_room" : 10,
	"Watch_TV" : 11,
	"Work" : 12
}
num_M = 51
num_I = 12
num_D = 15
num_T = 5
num_P = 1
num_E = 2
num_L = 11
first_M = 0
first_I = num_M
first_D = first_I + num_I
first_T = first_D + num_D
first_P = first_T + num_T
first_E = first_P + num_P
first_L = first_E + num_E
sensor_first_indices = {
	"M" : first_M,
	"I" : first_I,
	"D" : first_D,
	"T" : first_T,
	"P" : first_P,
	"E" : first_E,
	"L" : first_L
}
num_sensors = num_M + num_I + num_D + num_T + num_P + num_E + num_L

'''
Preprocesses the activity. For now, we don't care which resident made the activity.
Returns a number corresponding to the activity.
'''
def encoded_activity(activity):
	# remove resident info first (either "R1" or "R2")
	activity = activity[3:]
	if activity in activity_encoder_map:
		return activity_encoder_map[activity]
	else:
		raise Exception('activity could not be encoded. activity was: ' + activity)


def count_sensors(write_to_file=True):
	num_M = 0
	num_I = 0
	num_D = 0
	num_T = 0
	num_P = 0
	num_E = 0
	num_L = 0
	with open(filename) as f:
		lines = f.read().splitlines()
	for i in range(len(lines)):
		line = lines[i]
		line_contents = line.split()
		sensor_type = line_contents[2][0]
		if sensor_type == "M":
			num_M = max(num_M, int(line_contents[2][1:]))
		elif sensor_type == "I":
			num_I = max(num_I, int(line_contents[2][1:]))
		elif sensor_type == "D":
			num_D = max(num_D, int(line_contents[2][1:]))
		elif sensor_type == "T":
			num_T = max(num_T, int(line_contents[2][1:]))
		elif sensor_type == "P":
			num_P = max(num_P, int(line_contents[2][1:]))
		elif sensor_type == "E":
			num_E = max(num_E, int(line_contents[2][1:]))
		elif sensor_type == "L":
			num_L = max(num_L, int(line_contents[2][1:]))
		else:
			print("SENSOR TYPE NOT KNOWN " + sensor_type)
	if write_to_file:
		with open("sensor_info.txt", "w") as out:
			out.write("M: " + str(num_M) + "\n")
			out.write("I: " + str(num_I)+ "\n")
			out.write("D: " + str(num_D)+ "\n")
			out.write("T: " + str(num_T)+ "\n")
			out.write("P: " + str(num_P)+ "\n")
			out.write("E: " + str(num_E) + "\n")
			out.write("L: " + str(num_L))
	return num_M, num_I, num_D, num_T, num_P, num_E

# count_sensors()


def process_state(sensor, state):
	sensor_type = sensor[0]
	sensor_index = sensor_first_indices[sensor_type] + int(sensor[1:]) - 1
	if state == "OFF" or state == "ABSENT" or state == "CLOSE":
		sensor_state = 0
	elif state == "ON" or state == "PRESENT" or state == "OPEN":
		sensor_state = 1
	else:
		try:
			sensor_state = float(state)
		except ValueError:
			# there's a sensor value called SET_OPER_FLAGS...not sure 
			#     what this is so i'm just setting it to 0 for now
			sensor_state = 0
	return sensor_index, sensor_state

'''
num_features is defined as (window_size * num_sensors * 2)
Create a matrix that is num_features x num_samples.

window_size = 20 events after activity begins

'''

with open(filename) as f:
	lines = f.read().splitlines()
num_samples = 0
activity_indices = []
# at the beginning all sensors are 0
states = np.zeros((len(lines), num_sensors))
prev_state = np.zeros(num_sensors)
for i in range(len(lines)):
	line = lines[i]
	line_contents = line.split()
	
	# modify the state
	sensor_index, sensor_state = process_state(line_contents[2], line_contents[3])
	current_state = np.copy(prev_state)
	current_state[sensor_index] = sensor_state

	states[i] = current_state
	prev_state = current_state

	# detect if an activity has begun
	if len(line_contents) == 6 and line_contents[5] == "begin":
		num_samples += 1
		activity_indices.append(i)
print("finished pre processing states")
# TODO save states as python pkl file to save time in the future

num_features = window_size * num_sensors
feature_matrix = np.zeros((num_samples, num_features))
label_vector = np.zeros(num_samples)

for i in range(num_samples):
	sample_index = activity_indices[i]
	data = lines[sample_index].split()

	# add activity to label vector
	activity = data[4]
	label_vector[i] = encoded_activity(activity)

	# add window_size features to feature_matrix
	features = np.zeros(0)

	for j in range(window_size):
		line = lines[sample_index + j].split()
		np.append(features,states[sample_index + j])

#TODO save feature matrix and label vector as python pkl file
print("finished constructing feature matrix and label vector")
