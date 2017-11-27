from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
import numpy as np

FEATURES_PATH = "data/lstm/features.npy"
LABELS_PATH = "data/lstm/labels.npy"

if __name__ == "__main__":
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
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

    data_dim = features.shape[2]
    num_outputs = labels.shape[2]
    num_hidden_states = 200#data_dim

    model = Sequential()
    model.add(LSTM(num_hidden_states, return_sequences=True, input_shape=(1, data_dim)))
    model.add(Dense(num_outputs, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=25,
              epochs=100, validation_data=(x_val, y_val))
