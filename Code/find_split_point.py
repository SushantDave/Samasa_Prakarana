from keras.models import load_model
from keras.layers import Input, LSTM, Dense, Bidirectional, Embedding, Reshape, Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import sys

def test_model(input_word, model, char2idx, maxlen):
    X_test = [[char2idx[c] for c in w] for w in input_word]
    wordlen = len(X_test)
    gap = maxlen - wordlen
    lgap = [[char2idx['*']]] * gap
    X_test.extend(lgap)
    X_test = np.array(X_test)
  
    test = X_test.reshape((-1, maxlen))
    res = model.predict(test, verbose=0)
    res = res.reshape((maxlen))

    res = res[0:wordlen]
    #print(res)
    origres = res
    
    for j in range(wordlen):
        if(res[j] >= 0.5):
            res[j] = 1
        else:
            res[j] = 0
            
    ires = res.astype(int)
    return ires

def get_indicator_string(dtest):
    model = load_model("models_keras/stage_1_model.keras", compile=False)
    fp = open("models_keras/stage_1_dict.txt", "r")
    char2idx = json.loads(fp.read())
    char2idx = dict(char2idx)
    fp.close()
    fp = open("models_keras/stage_1_maxlen.txt", "r")
    maxlen = fp.read()
    maxlen = int(maxlen)
    fp.close()
    
    ires = test_model(dtest, model, char2idx, maxlen)
    return ires

if __name__=="__main__":
    word = sys.argv[1]
    arr = get_indicator_string(word)
    print(arr)
    for ctr in range(len(word)):
        if arr[ctr] == 0:
            print(word[ctr], end='')
        else:
            print(" "+word[ctr], end='')
