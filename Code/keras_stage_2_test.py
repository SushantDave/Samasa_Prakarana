from keras.models import Model
from keras.layers import Input
from tqdm import tqdm
import numpy as np

def decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    num_tokens = len(reverse_target_char_index)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, token_index['&']] = 1.

    # Sampling loop for a batch of sequences (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '$' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def infer_sandhi_split(dtest, model, token_index, max_encoder_seq_length, max_decoder_seq_length):
    input_texts = []
    target_texts = []

    latent_dim = 128
    encoder_inputs = model.input[0]  # input_1
    state_h_enc = model.layers[3].output
    state_c_enc = model.layers[4].output
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(2*latent_dim,))
    decoder_state_input_c = Input(shape=(2*latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[6]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    for data in dtest:
        [input_text, target_text] = data.split(',')
        input_text = input_text.strip()
        target_text = target_text.strip()
        input_texts.append(input_text)
        target_texts.append(target_text)

    num_tokens = len(token_index)
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_tokens), dtype='float32')

    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            if char not in token_index:
                continue
            encoder_input_data[i, t, token_index[char]] = 1.
        encoder_input_data[i, t + 1:, token_index['*']] = 1.

    # Reverse-lookup token index to decode sequences back to something readable.
    reverse_target_char_index = dict((i, char) for char, i in token_index.items())

    total = len(encoder_input_data)
    passed = 0
    results = []

    for idx in range(len(encoder_input_data)):
        # Take one sequence (part of the training set) for trying out decoding.
        input_seq = encoder_input_data[idx:idx + 1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, reverse_target_char_index)
        decoded_sentence = decoded_sentence.strip()
        decoded_sentence = decoded_sentence.strip('$')
        results.append(decoded_sentence)
        if decoded_sentence == target_texts[idx]:
            passed = passed + 1
        print("\rPassed: "+str(passed)+'/'+str(idx)+"(out of "+ str(total)+"), "+str(passed*100/(idx+1)), end='', flush=True)

from keras.models import load_model

model = load_model("models_keras/stage_2_model.keras")

import json

fp = open("models_keras/stage_2_dict.txt", "r")
token_index = dict(json.loads(fp.read()))
fp.close()
fp = open("models_keras/stage_2_maxenclen.txt", "r")
max_encoder_seq_length = int(fp.read())
fp.close()
fp = open("models_keras/stage_2_maxdeclen.txt", "r")
max_decoder_seq_length = int(fp.read())
fp.close()
#fp = open("models_keras/stage_2_dtest.txt", "r")
fp = open("sandhi_split_data_test.txt", "r")
dtest = fp.readlines()
fp.close()

infer_sandhi_split(dtest, model, token_index, max_encoder_seq_length, max_decoder_seq_length)
