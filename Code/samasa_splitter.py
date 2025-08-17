from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from find_split_point import get_indicator_string
from split_sandhi import sandhi_split
import sys
import numpy as np
import json
import random

def find_sandhi_window(indstr, onepos):
    start = max([0,onepos-2])
    end = min([len(indstr),onepos+3])
    return start,end

def handle_overlap_cases(str1, str2):
    # Case 1: str1 suffix matches str2 prefix
    max_n = min(len(str1), len(str2))
    for n in range(max_n, 0, -1):
        if str1[-n:] == str2[:n]:
            return str1[:-n] + str2

    # Case 2: str2 is a substring of str1
    if str2 in str1:
        i = str1.index(str2)
        return str1[:i + len(str2)]

    # Case 3: str1 is a substring of str2
    if str1 in str2:
        i = str2.index(str1)
        return str2[i:]

    # Case 4: str1 prefix matches str2 suffix
    for n in range(min(len(str1), len(str2)), 0, -1):
        if str1[:n] == str2[-n:]:
            return str1[:n]

    # If none of the cases match
    return str1+str2


def samasa_split(inp_str):
    # floating point values of indicator string
    findstr = get_indicator_string(inp_str)
    # integer values of indicator string
    indstr = []
    for j in range(len(findstr)):
        if(findstr[j] >= 0.5):
            indstr.append(1)
        else:
            indstr.append(0)

    onespos = np.where(indstr)[0]

    if len(onespos) == 0:
        maxidx = np.argmax(findstr)
        indstr[maxidx] = 1

    onespos = np.where(indstr)[0]

    outwords = []
    ppost = ""
    start_words_used = 0
    last_one_pos = 0
    last_sandhi_window_end = 0
    for oneidx, onepos in enumerate(onespos):
        # Original word starts from lastpos to onepos
        cur_sandhi_window_start, cur_sandhi_window_end = find_sandhi_window(indstr, onepos)
        out = sandhi_split(inp_str[cur_sandhi_window_start:cur_sandhi_window_end])
        [npre, npost] = out.split('-', 1)
        mid_word = inp_str[last_sandhi_window_end:cur_sandhi_window_start]
        overlap = cur_sandhi_window_start - last_sandhi_window_end

        if overlap == 0:
            outword = ppost + npre
        elif overlap < 0:
            outword = handle_overlap_cases(ppost, npre)
        else:
            outword = ppost + mid_word + npre

        ppost = npost
        last_sandhi_window_end = cur_sandhi_window_end
        last_one_pos = onepos
        outwords.append(outword)

    mid_word = inp_str[last_sandhi_window_end:]
    outword = ppost + mid_word
    outwords.append(outword)

    outstr = "-".join(outwords)

    return outstr

if __name__=="__main__":
    with open(sys.argv[1]) as fp:
        lines = fp.readlines()

    pos = 0
    pwords = 0
    twords = 0
    total = len(lines)
    fp = open("failed.txt", "w")
    for idx, line in enumerate(lines):
        [inp, out, _] = line.split(',')
        res = samasa_split(inp.strip())
        [r1, r2] = res.rsplit('-', 1)
        [o1, o2] = out.rsplit('-', 1)
        if r1 == o1 and r2[0] == o2[0]:
            pos = pos + 1
        else:
            fp.write(inp+","+out+","+res+"\n")
            fp.flush()

        reswords = r1.split('-')
        outwords = o1.split('-')
        for outword in outwords:
            twords = twords + 1
            if outword in reswords:
                pwords = pwords + 1

        twords = twords + 1
        if r2[0] == o2[0]:
            pwords = pwords + 1

        print("\r[Words passed: "+str(pos)+'/'+str(idx+1)+"(out of "+ str(total)+"), "+str(pos*100/(idx+1)) + "] [Components passed: " + str(pwords)+"/"+str(twords)+" "+str(pwords*100/twords)+"]", end='', flush=True)
    fp.close()

    print(str(pos)+"/"+str(len(lines)))
    print(str(pwords)+"/"+str(twords))
