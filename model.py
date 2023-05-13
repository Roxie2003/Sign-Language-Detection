import os
import numpy as np

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

def create_model():
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_DataFinal')

    # Actions that we try to detect
    actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Hello','Yes','Okay','I Love you'])

    # Thirty videos worth of data
    no_sequences = 10

    # Videos are going to be 15 frames in length
    sequence_length = 15

    label_map = {label: num for num, label in enumerate(actions)}

    # Loading frames data from files for every action
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    x = np.array(sequences)
    y = to_categorical(labels).astype(int)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir,)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 126)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    #model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #model.fit(x, y, epochs=125, callbacks=[tb_callback])
    #model.save('action4.h5')

    model.load_weights('allActionsA-Z.h5')

    return model