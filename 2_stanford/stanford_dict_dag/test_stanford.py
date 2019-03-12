# _*_coding:utf-8_*_

from __future__ import print_function
import time
from stanfordcorenlp import StanfordCoreNLP

local_corenlp_path = '/opt/typ/data/stanford-corenlp-full-2018-02-27/'

# Simple usage
# nlp = StanfordCoreNLP(local_corenlp_path, quiet=False)
#
# sentence = 'Guangdong University of Foreign Studies (GDUFS) is located in Guangzhou.'
# print('Tokenize:', nlp.word_tokenize(sentence))
# print('Part of Speech:', nlp.pos_tag(sentence))
# print('Named Entities:', nlp.ner(sentence))
# print('Constituency Parsing:', nlp.parse(sentence))
# print('Dependency Parsing:', nlp.dependency_parse(sentence))
#
# nlp.close()

# Other human languages support, e.g. Chinese
sentence = '清华大学位于北京。'

nlp=StanfordCoreNLP(local_corenlp_path, lang='zh', quiet=False)
while True:
    sentence=input("请输入测试语句：")
    time1 = time.time()
    cut=nlp.word_tokenize(sentence)
    print("分词：耗时",time.time()-time1,cut)
    # print("词性标注：",nlp.pos_tag(sentence))
    # print("命名实体识别：",nlp.ner(sentence))
    # print("句法分析：",nlp.parse(sentence))
    # print("语义依存分析：",nlp.dependency_parse(sentence))

# General Stanford CoreNLP API
# nlp = StanfordCoreNLP(local_corenlp_path, memory='8g', lang='zh')
# print(nlp.annotate(sentence))
# nlp.close()

# nlp = StanfordCoreNLP(local_corenlp_path)
# text = 'Guangdong University of Foreign Studies is located in Guangzhou. ' \
#        'GDUFS is active in a full range of international cooperation and exchanges in education. '
# pros = {'annotators': 'tokenize,ssplit,pos', 'pinelineLanguage': 'en', 'outputFormat': 'xml'}
# print(nlp.annotate(text, properties=pros))
# nlp.close()
#
# # Use an existing server
# nlp = StanfordCoreNLP('http://corenlp.run', port=80)
