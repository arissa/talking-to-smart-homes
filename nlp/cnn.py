from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
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

    # data_dim = features.shape[1]
    # num_outputs = labels.shape[1]

    data_dim = features.shape[2]
    num_outputs = labels.shape[2]

    # hyperparameters
    num_hidden_states = 200#data_dim
    num_filters = 1
    filter_size = 1
    batch_size = 25
    num_epochs = 5000


    model = Sequential()
    model.add(Convolution1D(num_filters, filter_size, activation='relu', input_shape=(1, data_dim)))
    # model.add(Convolution1D(num_filters, filter_size, activation='relu'))
    model.add(Dense(num_outputs, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=num_epochs, validation_data=(x_val, y_val))
