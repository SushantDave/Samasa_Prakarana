from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
import json
import random

def train_sandhi_split(dtrain, dtest):
    batch_size = 64  # Batch size for training.
    epochs = 90  # Number of epochs to train for.
    latent_dim = 128  # Latent dimensionality of the encoding space.

    # Vectorize the data.
    input_texts = []
    target_texts = []
    X_tests = []
    Y_tests = []
    characters = set()

    for data in dtrain:
        [input_text, target_text] = data.split(',')
        input_text = input_text.strip()
        target_text = target_text.strip()

        # We use "&" as the "start sequence" character for the targets, and "$" as "end sequence" character.
        target_text = '&' + target_text + '$'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in characters:
                characters.add(char)
        for char in target_text:
            if char not in characters:
                characters.add(char)

    for data in dtest:
        [input_text, target_text] = data.split(',')
        input_text = input_text.strip()
        target_text = target_text.strip()

        # We use "&" as the "start sequence" character for the targets, and "$" as "end sequence" character.
        target_text = '&' + target_text + '$'
        X_tests.append(input_text)
        Y_tests.append(target_text)
        for char in input_text:
            if char not in characters:
                characters.add(char)
        for char in target_text:
            if char not in characters:
                characters.add(char)

    # Using '*' for padding
    characters.add('*')

    characters = sorted(list(characters))
    num_tokens = len(characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique tokens:', num_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    token_index = dict([(char, i) for i, char in enumerate(characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_tokens), dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_tokens), dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, token_index[char]] = 1.
        encoder_input_data[i, t + 1:, token_index['*']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, token_index[char]] = 1.
        decoder_input_data[i, t + 1:, token_index['*']] = 1.
        decoder_target_data[i, t:, token_index['*']] = 1.

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_tokens), name='stage2_encinp')
    encoder_bilstm = Bidirectional(LSTM(latent_dim, return_state=True, dropout=0.5))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bilstm(encoder_inputs)
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_tokens), name='stage2_decinp')

    # We set up our decoder to return full output sequences, and to return internal states as well.
    # We dont use the return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True, dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)

    return (model, token_index, max_encoder_seq_length, max_decoder_seq_length)

with open("sandhi_split_data_train.txt", 'r', encoding='utf-8') as f:
#with open("./Samasa_Prakarana/Data/samasa_split_train_data.csv", 'r', encoding='utf-8') as f:
    dl = f.readlines()
random.shuffle(dl)
dtrain, dtest = train_test_split(dl, test_size=0.2, random_state=1)
(model, token_index, max_encoder_seq_length, max_decoder_seq_length) = train_sandhi_split(dtrain, dtest)
model.save("models_keras/stage_2_model.keras")

fp = open("models_keras/stage_2_dict.txt", "w")
fp.write(json.dumps(token_index))
fp.close()
fp = open("models_keras/stage_2_maxenclen.txt", "w")
fp.write(str(max_encoder_seq_length))
fp.close()
fp = open("models_keras/stage_2_maxdeclen.txt", "w")
fp.write(str(max_decoder_seq_length))
fp.close()
fp = open("models_keras/stage_2_dtest.txt", "w")
fp.write(json.dumps(dtest))
fp.close()
