import json

import numpy as np
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config


def load_data(filename):
    """Read processed data from file and return data ready for training."""
    with open(filename) as data_file:
        database = json.load(data_file)
    input_seqs = database['data']
    input_labels = database['labels']
    output_labels = []
    data_dict = {}
    for index, item in enumerate(input_seqs):
        length = len(item)
        for l in range(1, length + 1):
            element = tuple(item[i] for i in range(l))
            if element in data_dict:
                if input_labels[index]:
                    data_dict[element][0] += 1  # first_choice_win
                data_dict[element][1] += 1  # count
            else:
                data_dict[element] = [1 if input_labels[index] else 0, 1]  # [first_choice_win, count]
    padded_sequences = pad_sequences(list(data_dict.keys()), padding='post')
    # Calculate win probability for each sequence and append to output labels.
    for sequence in data_dict:
        win_count_list = data_dict[sequence]
        output_labels.append(win_count_list[0] / win_count_list[1])
    np_labels = np.array(output_labels)
    return padded_sequences, np_labels


def load_data_including_duplicates(filename):
    """Read processed data from file and return data ready for training."""
    with open(filename) as data_file:
        database = json.load(data_file)
    input_seqs = database['data']
    input_labels = database['labels']
    output_seqs = []
    output_labels = []
    for index, item in enumerate(input_seqs):
        length = len(item)
        for l in range(1, length + 1):
            element = tuple(item[i] for i in range(l))
            output_seqs.append(element)
        output_labels.extend(length * [1 if input_labels[index] else 0])
    padded_sequences = pad_sequences(output_seqs, padding='post')
    # Calculate win probability for each sequence and append to output labels.
    np_labels = np.array(output_labels)
    return padded_sequences, np_labels


def load_data_full_drafts():
    with open(config.TRAINING_DATA_FILE) as data_file:
        database = json.load(data_file)
    np_sequences = np.array(database['data'])
    np_labels = np.array([1 if i else 0 for i in database['labels']])
    return np_sequences, np_labels


def split_data(sequences, labels, training_fraction=0.9):
    """Split data into training and testing parts."""
    training_rows = round(training_fraction * (labels.shape[0]))
    sequences_train = sequences[:training_rows]
    labels_train = labels[:training_rows]
    sequences_test = sequences[training_rows:]
    labels_test = labels[training_rows:]
    return sequences_train, labels_train, sequences_test, labels_test


def build_model(hero_count, draft_length):
    model = Sequential()

    model.add(Embedding(hero_count + 1, 6, input_length=draft_length))
    # model.add(LSTM(50, dropout=0.2, return_sequences=True))
    model.add(LSTM(50, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # sequence, label = load_data_including_duplicates(config.TRAINING_DATA_FILE)
    # seq_train, lab_train, seq_test, lab_test = split_data(sequence, label)
    seq_train, lab_train = load_data_full_drafts()
    print(seq_train[0].shape)
    model = build_model(115, config.DRAFT_LENGTH)
    model.summary()
    model.fit(seq_train, lab_train, batch_size=128, epochs=20, validation_split=0.1)

    # https://www.dotabuff.com/matches/4048084806
    test_array = np.array([[99, 70, 17, 4, 22, 68, 83, 36, 5, 97, 74, 59, 81, 61, 92, 43, 26, 41, 67, 6, 37, 42]])

    first_choice_win_confidence = model.predict(test_array)[0, 0]
    print(first_choice_win_confidence)
