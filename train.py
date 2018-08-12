import json
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, GRU, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config


def load_data(filename):
    """Read processed data from file and return data ready for training.

    No duplicate sequences. Labels are probabilities.
    """
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
                data_dict[element] = [int(input_labels[index]), 1]  # [first_choice_win, count]
    padded_sequences = pad_sequences(list(data_dict.keys()), padding='post')
    # Calculate win probability for each sequence and append to output labels.
    for sequence in data_dict:
        win_count_list = data_dict[sequence]
        output_labels.append(win_count_list[0] / win_count_list[1])
    np_labels = np.array(output_labels)
    return padded_sequences, np_labels


def load_data_duplicates(filename):
    """Read processed data from file and return data ready for training.

    Duplicate sequences with labels either 0 or 1.
    """
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
        output_labels.extend(length * [int(input_labels[index])])
    padded_sequences = pad_sequences(output_seqs, padding='post')
    np_labels = np.array(output_labels)
    return padded_sequences, np_labels


def load_data_picks_duplicates(filename):
    """Read processed data from file and return data ready for training.

    Picks only (sequence length 10). Duplicates included.
    """
    with open(filename) as data_file:
        database = json.load(data_file)
    input_seqs = database['data']
    input_labels = database['labels']
    output_seqs = []
    output_labels = []
    pick_indices = [6, 7, 8, 9, 14, 15, 16, 17, 20, 21]
    for index, item in enumerate(input_seqs):
        for l in range(1, 11):
            element = tuple(item[i] for i in pick_indices[:l])
            output_seqs.append(element)
        output_labels.extend(10 * [int(input_labels[index])])
    padded_sequences = pad_sequences(output_seqs, padding='post')
    np_labels = np.array(output_labels)
    return padded_sequences, np_labels


def load_data_full_drafts(filename):
    """Read processed data from file and return data ready for training.

    Full draft sequences only (no zeros). Duplicates included.
    """
    with open(filename) as data_file:
        database = json.load(data_file)
    np_sequences = np.array(database['data'])
    np_labels = np.array([int(i) for i in database['labels']])
    return np_sequences, np_labels


def load_data_full_drafts_picks_only(filename):
    """Read processed data from file and return data ready for training.

    Full draft sequences with picks only (no zeros, length 10). Duplicates included.
    """
    with open(filename) as data_file:
        database = json.load(data_file)
    pick_indices = [6, 7, 8, 9, 14, 15, 16, 17, 20, 21]
    sequences = [[s[i] for i in pick_indices] for s in database['data']]
    np_sequences = np.array(sequences)
    np_labels = np.array([int(i) for i in database['labels']])
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

    model.add(Embedding(hero_count, 4, input_length=draft_length, mask_zero=False))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, unroll=False, return_sequences=True))
    # model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, unroll=False))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, unroll=False, reset_after=True, return_sequences=True))
    # model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, unroll=False, implementation=1, reset_after=True))
    model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2, unroll=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == '__main__':
    sequence, label = load_data_picks_duplicates(config.TRAINING_DATA_FILE)
    seq_train, lab_train, seq_test, lab_test = split_data(sequence, label)
    model = build_model(115, config.DRAFT_LENGTH)
    # model = build_model(115, 10)
    model.summary()
    early_stop = EarlyStopping('val_loss', patience=5, mode='min')
    history = model.fit(seq_train, lab_train, batch_size=512, epochs=100, validation_split=0.1, callbacks=[early_stop])

    # Graph training and validation loss and accuracy.
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()  # clear figure
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Evaluate model on test data.
    results = model.evaluate(seq_test, lab_test)
    print(results)

    # https://www.dotabuff.com/matches/4053143871
    # test_array = np.array([[99, 92, 5, 85, 32, 101, 26, 58, 21, 111, 4, 56, 61, 59, 20, 6, 67, 63, 47, 22, 9, 106]])
    test_array = np.array([[26, 58, 21, 111, 20, 6, 67, 63, 9, 106]])
    first_choice_win_confidence = model.predict(test_array)[0, 0]
    print(first_choice_win_confidence)

    first_choice = bool(input('You are first pick (1 or 0): '))
    current_choices = np.zeros((1, 10))
    for i in range(10):
        with open('heroes.json') as hero_file:
            heroes = json.load(hero_file)['heroes']
        for h in heroes:
            current_choices[0, i] = h['id']
            h['first_choice_win'] = model.predict(current_choices)[0, 0]
        sorted_heroes = sorted(heroes, key=itemgetter('first_choice_win'), reverse=first_choice)
        counter = 0
        for h in sorted_heroes:
            print('{:3} {:20} {:6.6}'.format(h['id'], h['localized_name'], str(h['first_choice_win'])))
            counter += 1
            if counter > 20:
                break
        current_choices[0, i] = int(input('Hero choice {}: '.format(i + 1)))
        print()
    confidence = model.predict(current_choices)[0, 0]
    if not first_choice:
        confidence = 1 - confidence
    print('Win probability: {}%'.format(str(round(100 * confidence, 3))))
