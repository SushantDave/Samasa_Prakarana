import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Embedding, Reshape, Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf

@keras.saving.register_keras_serializable(package="my_package", name="custom_loss")
def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    loss1 = mse(y_true, y_pred)
    pred_mean = tf.reduce_mean(y_pred)
    true_mean = tf.cast(tf.reduce_mean(y_true), tf.float32)
    loss2 = (pred_mean - true_mean)**2
    loss3 = tf.reduce_sum(tf.square(tf.subtract(tf.cast(y_true, tf.float32), y_pred)))
    loss = loss3 #loss1 + 0.5*loss2
    return loss

def train_predict_sandhi_window(dtrain, dtest):
    batch_size = 64  # Batch size for training.
    epochs = 200  # Number of epochs to train for.
    latent_dim = 64  # Latent dimensionality of the encoding space.

    # Vectorize the data.
    inputs = []
    targets = []
    characters = set()

    for data in dtrain:
        target = np.array(list(data[1]))
        input_word = data[0]

        inputs.append(input_word)
        targets.append(target)

        for char in input_word:
            if char not in characters:
                characters.add(char)

    maxlen = max([len(s) for s in inputs])
    print(inputs[0])
    print(maxlen)

    """
    * is used as padding character
    """
    characters.add('*')
    char2idx = dict([(char, i) for i, char in enumerate(characters)])
    num_tokens = len(characters)
    
    X_train = [[char2idx[c] for c in w] for w in inputs]
    X_train = pad_sequences(maxlen=maxlen, sequences=X_train, padding="post", value=char2idx['*'])
    
    Y_train = targets
    Y_train = pad_sequences(maxlen=maxlen, sequences=Y_train, padding="post", value=0.0)
    Y_train = np.array(Y_train).reshape(-1, maxlen, 1)
    
    inputs = []
    targets = []
    for data in dtest:
        target = np.array(list(data[1]))
        input_word = data[0]
    
        inputs.append(input_word)
        targets.append(target)
    
        for char in input_word:
            if char not in characters:
                characters.add(char)

    print('Number of training samples:', len(X_train))
    print('Number of unique tokens:', num_tokens)

    # Define an input sequence and process it.
    inputword = Input(shape=(maxlen,))
    embed = Embedding(input_dim=num_tokens, output_dim=8, input_length=maxlen, mask_zero=True)(inputword)
    bilstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    out, forward_h, forward_c, backward_h, backward_c = bilstm(embed)
    outd = Dropout(0.5)(out)
    outputtarget = Dense(1, activation="sigmoid")(outd)

    model = Model(inputword, outputtarget)
    #optimizer = tf.keras.optimizers.Adam(0.001)
    #optimizer.learning_rate.assign(0.01)
    model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train, batch_size, epochs, validation_split=0.1)
    return model, char2idx, maxlen

with open("./Samasa_Prakarana/Data/final_data_slp1.csv", 'r', encoding='utf-8') as f:
    odl = f.readlines()
dl = []
for ol in odl:
    lol = ol.split(',')
    dl.append([lol[0], lol[2]])

dtrain, dtest = train_test_split(dl, test_size=0.2, random_state=1)
model, char2idx, maxlen = train_predict_sandhi_window(dtrain, dtest)
model.save("models_keras/stage_1_model.keras")

import json

fp = open("models_keras/stage_1_dict.txt", "w")
fp.write(json.dumps(char2idx))
fp.close()
fp = open("models_keras/stage_1_maxlen.txt", "w")
fp.write(str(maxlen))
fp.close()
fp = open("models_keras/stage_1_dtest.txt", "w")
fp.write(json.dumps(dtest))
fp.close()
