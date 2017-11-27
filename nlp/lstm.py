from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np

FEATURES_PATH = "data/features.npy"
LABELS_PATH = "data/labels.npy"

if __name__ == "__main__":
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    # labels = labels.reshape((labels.shape[0], 1))
    print("FEATURES SHAPE")
    print(features.shape)
    print("LABELS SHAPE")
    print(labels.shape)
    percent_train = 0.8
    num_train = int(0.8 * features.shape[0])
    x_train = features[:num_train, :]
    y_train = labels [:num_train]
    x_val = features[num_train:, :]
    y_val = labels[num_train:] 
    print("X TRAIN")
    print(x_train.shape)
    print("Y TRAIN")
    print(y_train.shape)

    # input_shape = (features.shape[1], 1)
    data_dim = features.shape[2]
    num_outputs = labels.shape[2]
    num_hidden_states = 128#data_dim

    model = Sequential()
    model.add(LSTM(num_hidden_states, return_sequences=True, input_shape=(1, data_dim)))
    model.add(Dense(num_outputs, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=1,
              epochs=8, validation_data=(x_val, y_val))
