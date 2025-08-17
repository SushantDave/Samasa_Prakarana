from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
import nltk
from tqdm import tqdm

#with open("Samasa_Prakarana/Data/parallel_data_iast.tsv", 'r', encoding='utf-8') as f:
with open("parallel_data_iast.tsv", 'r', encoding='utf-8') as f:
    itrans_list = f.readlines()

error_cnt = 0
io_list = []
print("Converting data to SLP1")
for line in tqdm(itrans_list):
    slp1_line = transliterate(line.strip(), sanscript.IAST, sanscript.SLP1)
    elements = slp1_line.split("\t")
    if(len(elements) != 3):
        error_cnt += 1
        print(line)
        if(error_cnt > 5):
            break
    in_words = elements[1].split()
    out_words = elements[2].split()

    #Remove duplicates in input and output sentences
    in_words, out_words = list(set(in_words) - set(out_words)), list(set(out_words) - set(in_words))
    io_list.append([in_words, out_words])

def find_min_dist_window(word2, word1):
    min_dist = 10000
    min_start_idx = 0
    min_word_len = 0
    for start_id in range(len(word1)-1):
        for window_len in range(1, len(word1)-start_id+1):
            dist = nltk.edit_distance(word2, word1[start_id:start_id + window_len])
            if(dist <= min_dist):
                min_dist = dist
                min_start_idx = start_id
                min_word_len = window_len
    return [min_dist, min_start_idx, min_word_len]

def find_min_dist_idx(word2, sen1):
    word2 = word2.replace('-', '')
    min_dist = 10000
    min_idx = 0
    for idx in range(len(sen1)):
        [dist, sid, wlen] = find_min_dist_window(word2, sen1[idx])
        if(dist < min_dist):
            min_dist = dist
            min_word = sen1[idx][sid:sid + wlen]
    return [min_dist, min_word]

final_io_list = []
print("Mapping the words")
for io in tqdm(io_list):
    sen1 = io[0]
    sen2 = io[1]
    for cnt in range(len(sen2)):
        if('-' in sen2[cnt]):
            [min_dist, min_word] = find_min_dist_idx(sen2[cnt], sen1)
            word1 = min_word
            word2 = sen2[cnt]
            word2 = word2.replace('-', '')
            dist = min_dist/max(len(word1), len(word2))
            final_io_list.append(word1+','+sen2[cnt]+','+('%.2f' % dist))

def find_mapping_len(word2, word1, start_id):
    min_dist = 10000
    min_word_len = 0
    for curr_id in range(start_id+1, len(word1)-1):
        dist = nltk.edit_distance(word2, word1[start_id:curr_id])
        if(dist <= min_dist):
            min_dist = dist
            min_word_len = curr_id - start_id
    return min_word_len

seq_data_list = []

# Presence of 1 marks a new word. Basically, 1 implies beginning of new word,
# (except for 1st word), with zero or more characters following it (denoted by 0)
print("Creating indicator string")
for data in tqdm(final_io_list):
    [word1, word2, dist] = data.split(',')
    nword2 = word2.replace('-','')
    if float(dist) == 0.00 or len(word1) == len(nword2):
        wlist = word2.split('-')
        #outword = '0'*(len(wlist[0])-1) + '1'
        outword = '0'*(len(wlist[0]))
        for widx in range(1, len(wlist)-1):
            #outword = outword + '1' + '0'*(len(wlist[widx])-2) + '1'
            outword = outword + '1' + '0'*(len(wlist[widx])-1)
        outword = outword + '1' + '0'*(len(wlist[-1])-1)
    else:
        wlist = word2.split('-')
        start_id = 0
        wlen = find_mapping_len(wlist[0], word1, start_id)
        #outword = '0'*(wlen-1) + '1'
        outword = '0'*(wlen)
        start_id = start_id + wlen
        for widx in range(1, len(wlist)-1):
            wlen = find_mapping_len(wlist[widx], word1, start_id)
            #outword = outword + '1' + '0'*(wlen-2) + '1'
            outword = outword + '1' + '0'*(wlen-1)
            start_id = start_id + wlen
        outword = outword + '1' + '0'*(len(word1)-start_id-1)

    seq_data_list.append([word1, word2, outword, dist])

print("Writing entries to file")
fp = open("final_data_slp1.csv", 'w', encoding='utf8')
for io in tqdm(seq_data_list):
    if(len(io[0]) != len(io[2])):
        print(io[0])
        break
    if(io[1].count('-') != io[2].count('1')):
        print(io[0])
        break
    fp.write(io[0]+','+io[1]+','+io[2]+'\n')
fp.close()

maxlen = 0
maxword = ""
with open("final_data_slp1.csv", 'r', encoding='utf8') as fp:
    iol = fp.readlines()

for io in iol:
    wd = io.split(',')[0]
    if(len(wd) > maxlen):
        maxlen = len(wd)
        maxword = wd
print(maxlen)
print(maxword)
