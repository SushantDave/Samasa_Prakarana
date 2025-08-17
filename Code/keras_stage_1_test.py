from keras.models import load_model
from keras.layers import Input, LSTM, Dense, Bidirectional, Embedding, Reshape, Dropout
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm

def is_nearby_one(ctr, ires):
    wlen = len(ires)
    if ires[ctr] == 1:
        return True
    if ctr > 0 and ires[ctr-1] == 1:
        return True
    if ctr < wlen-1 and ires[ctr] == 1:
        return True
    return False

def compare_samasa_approx(ires, iact):
    found = 0
    wlen = len(iact)
    for ctr in range(wlen):
        if iact[ctr] == 1:
            if is_nearby_one(ctr, ires):
                found = found + 1
    return found

def test_model(dtest, model, char2idx, maxlen):
    np.set_printoptions(precision=2, suppress=True)
    passed = 0
    failed = 0
    tp = 0
    fp = 0
    fn = 0
    total_samasa = 0
    correct_samasa = 0
    approx_correct = 0
    inputs = []
    targets = []
    for data in dtest:
        target = np.array(list(data[1]))
        input_word = data[0]
    
        inputs.append(input_word)
        targets.append(target)
    
    X_test = [[char2idx[c] for c in w] for w in inputs]
    X_test = pad_sequences(maxlen=maxlen, sequences=X_test, padding="post", value=char2idx['*'])
    
    Y_test = targets
    Y_test = pad_sequences(maxlen=maxlen, sequences=Y_test, padding="post", value=0.0)
    Y_test = np.array(Y_test).reshape(-1, maxlen, 1)
   
    startlist = []
    filp = open("failed.txt", 'w')
    for i in tqdm(range(X_test.shape[0])):
        test = X_test[i].reshape((-1, maxlen))
        res = model.predict(test, verbose=0)
        res = res.reshape((maxlen))
        act = Y_test[i].reshape((maxlen))


        wordlen = 0
        for j in range(maxlen):
            if X_test[i][j] == char2idx['*']:
                break
            else:
                wordlen = wordlen + 1

        res = res[0:wordlen]
        act = act[0:wordlen]
        origres = np.copy(res)
        
        for j in range(wordlen):
            if(res[j] >= 0.1):
                res[j] = 1
            else:
                res[j] = 0
                
        ires = res.astype(int)
        iact = act.astype(int)

        for idx, val in enumerate(ires):
            if val == 1 and iact[idx] == 1:
                tp = tp + 1
            elif val == 1 and iact[idx] == 0:
                fp = fp + 1
            elif val == 0 and iact[idx] == 1:
                fn = fn + 1

        comparison = ires == iact
                
        if comparison.all():
            passed = passed + 1
        else:
            failed = failed + 1
            filp.write(str(list(dtest[i][0])))
            filp.write('\n')
            filp.write(str(list(origres)))
            filp.write('\n')
            filp.write(str(list(ires)))
            filp.write('\n')
            filp.write(str(list(iact)))
            filp.write('\n')
            filp.write('*****************************************************\n')

    filp.close()
    print("Words passed: " + str(passed))
    print("Words failed: " + str(failed))
    print("Words total: " + str(failed + passed))
    print("Word pass rate: " + str(passed*100/(passed+failed)))
    print("True Positive: " + str(tp))
    print("False Positive: " + str(fp))
    print("False Negative: " + str(fn))
    print("Precision: " + "{:.3f}".format(tp/(tp+fp)))
    print("Recall: " + "{:.3f}".format(tp/(tp+fn)))
    print("F1: " + "{:.3f}".format(2*tp/(2*tp+fp+fn)))

    return startlist

model = load_model("models_keras/stage_1_model.keras", compile=False)

import json
fp = open("models_keras/stage_1_dict.txt", "r")
char2idx = json.loads(fp.read())
char2idx = dict(char2idx)
fp.close()
fp = open("models_keras/stage_1_maxlen.txt", "r")
maxlen = fp.read()
maxlen = int(maxlen)
fp.close()
fp = open("models_keras/stage_1_dtest.txt", "r")
dtest = json.loads(fp.read())
dtest = list(dtest)
fp.close()

test_model(dtest, model, char2idx, maxlen)
