# -*- coding: utf-8 -*-
import sys, getopt
import re
from math import log

from stanfordcorenlp import StanfordCoreNLP

local_corenlp_path = '/opt/typ/data/stanford-corenlp-full-2018-02-27/'


class Segmentation:
    re_eng = re.compile('[a-zA-Z0-9]', re.U)
    wfreq = {}
    total = 0

    def __init__(self, dict_path):
        with open(dict_path, "rb") as f:
            count = 0
            for line in f:
                try:
                    line = line.strip().decode('utf-8')
                    word, freq = line.split(',')[:2]
                    freq = int(freq)
                    self.wfreq[word] = freq
                    for idx in range(len(word)):
                        wfrag = word[:idx + 1]
                        if wfrag not in self.wfreq:
                            self.wfreq[wfrag] = 0  # trie: record char in word path
                    self.total += freq
                    count += 1
                except Exception as e:
                    print("%s add error!" % line)
                    print(e)
                    continue
        print("load dict: %d" % count)

    def seg(self, sentence):
        #         sentence = sentence.strip()
        # print("sentence", sentence)
        DAG = self.get_DAG(sentence)
        # print(DAG)
        route = {}
        self.get_route(DAG, sentence, route)
        # print(route)
        x = 0
        N = len(sentence)
        buf = ''
        lseg = []
        while x < N:
            y = route[x][1] + 1
            l_word = ''.join(sentence[x:y])
            if self.re_eng.match(l_word) and len(l_word) == 1:
                buf += l_word
                x = y
            else:
                if buf:
                    lseg.append(buf + " ")
                    buf = ''
                lseg.append(l_word)
                x = y
        if buf:
            lseg.append(buf + " ")
        return lseg

    def get_route(self, DAG, sentence, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((log(self.wfreq.get(''.join(sentence[idx:x + 1])) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])

    def get_DAG(self, sentence):
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]

            while i < N and frag in self.wfreq:
                if self.wfreq[frag]:
                    tmplist.append(i)
                i += 1
                frag = ''.join(sentence[k:i + 1])
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG


if __name__ == "__main__":
    seger = Segmentation("/opt/gongxf/python3_pj/nlp_practice/1_word_ltp_stanford/stanford-corenlp/new_dict.txt")
    # seger.seg('任性付我们是任性贷，你们什么时候用')
    # sentence='任性付我们是任性贷，你们什么时候用实名认证'
    nlp=StanfordCoreNLP(local_corenlp_path, lang='zh', quiet=False)
    while True:
        sentence=input("请输入测试语句：")
        stanford_cut=nlp.word_tokenize(sentence)
        print("stanford_cut",stanford_cut)
        cut=seger.seg(stanford_cut)
        print("cut",cut)
