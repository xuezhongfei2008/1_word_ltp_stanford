# _*_coding:utf-8_*_

from __future__ import print_function
import time
from stanfordcorenlp import StanfordCoreNLP

local_corenlp_path = '/opt/typ/data/stanford-corenlp-full-2018-02-27/'


# 分词
def vocab_tokenize(nlp, sentence):
    pros = {'annotators': 'tokenize',
            'tokenize.language': 'zh',
            'segment.model': 'edu/stanford/nlp/models/segmenter/chinese/ctb.gz',
            'segment.sighanCorporaDict': 'edu/stanford/nlp/models/segmenter/chinese',
            # 'segment.serDictionary':'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz,edu/stanford/nlp/models/segmenter/chinese/only_word.txt',
            # 'segment.serDictionary':'edu/stanford/nlp/models/segmenter/chinese/only_word.txt',
            'segment.serDictionary': 'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz',
            'outputFormat': 'text'}
    cut = nlp.annotate(sentence, properties=pros)
    # cut=nlp.word_tokenize(sentence)
    print("分词：", cut)


# 词性标注
def vocab_pos(nlp, sentence):
    pros = {'annotators': 'pos',
            'pos.model': 'edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger',
            'tokenize.language': 'Whitespace',
            'outputFormat': 'text'}
    pos = nlp.annotate(sentence, properties=pros)
    # pos=nlp.pos_tag(sentence)
    print("词性标注：", pos)


# 实体识别
def vocab_ner(nlp, sentence):
    pros = {'annotators': 'ner',
            'tokenize.language': 'Whitespace',
            # 'segment.model': 'edu/stanford/nlp/models/segmenter/chinese/ctb.gz',
            # 'segment.sighanCorporaDict': 'edu/stanford/nlp/models/segmenter/chinese',
            # 'segment.serDictionary': 'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz',
            # 'segment.sighanPostProcessing': 'true',

            'ner.model': 'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz',
            'ner.language': 'chinese',
            'entitylink.wikidict': 'edu/stanford/nlp/models/kbp/wikidict_chinese.tsv.gz',
            'ner.applyNumericClassifiers': 'false',
            'ner.useSUTime': 'false',
            'regexner.mapping': 'edu/stanford/nlp/models/kbp/cn_regexner_mapping.tab',
            'regexner.validpospattern': '^(NR|NN|JJ).*',
            'regexner.ignorecase': 'true',
            'regexner.noDefaultOverwriteLabels': 'CITY',
            'outputFormat': 'text'}
    ner = nlp.annotate(sentence, properties=pros)
    # print("命名实体识别：",nlp.ner(sentence))
    print("实体提取标注：", ner)
    # print("句法分析：",nlp.parse(sentence))
    # print("语义依存分析：",nlp.dependency_parse(sentence))


# 指代消解
def vocab_coref(nlp, sentence):
    pros = {
        # Pipeline options - lemma is no-op for Chinese but currently needed because coref
        # demands it (bad old requirements system)
        'annotators': 'tokenize, ssplit, pos, lemma, ner, parse, coref',

        # segment
        # tokenize.language' : 'zh'
        'tokenize.language': 'Whitespace',
        'segment.model': 'edu/stanford/nlp/models/segmenter/chinese/ctb.gz',
        'segment.sighanCorporaDict': 'edu/stanford/nlp/models/segmenter/chinese',
        'segment.serDictionary': 'edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz',
        'segment.sighanPostProcessing': 'true',

        # sentence split
        'ssplit.boundaryTokenRegex': '[.。,]|[!?！？]+',

        # pos
        'pos.model': 'edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger',

        # ner
        'ner.language': 'chinese',
        'ner.model': 'edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz',
        'ner.applyNumericClassifiers': 'true',
        'ner.useSUTime': 'false',

        # regexner
        'regexner.mapping': 'edu/stanford/nlp/models/kbp/cn_regexner_mapping.tab',
        'regexner.validpospattern': '^(NR|NN|JJ).*',
        'regexner.ignorecase': 'true',
        'regexner.noDefaultOverwriteLabels': 'CITY',

        # parse
        'parse.model': 'edu/stanford/nlp/models/srparser/chineseSR.ser.gz',

        # depparse
        'depparse.model': 'edu/stanford/nlp/models/parser/nndep/UD_Chinese.gz',
        'depparse.language': 'chinese',

        # coref
        'coref.sieves': 'ChineseHeadMatch, ExactStringMatch, PreciseConstructs, StrictHeadMatch1, StrictHeadMatch2, StrictHeadMatch3, StrictHeadMatch4, PronounMatch',
        'coref.input.type': 'raw',
        'coref.postprocessing': 'true',
        'coref.calculateFeatureImportance': 'false',
        'coref.useConstituencyTree': 'true',
        'coref.useSemantics': 'false',
        'coref.algorithm': 'hybrid',
        'coref.path.word2vec': '',
        'coref.language': 'zh',
        'coref.defaultPronounAgreement': 'true',
        'coref.zh.dict': 'edu/stanford/nlp/models/dcoref/zh-attributes.txt.gz',
        'coref.print.md.log': 'false',
        'coref.md.type': 'RULE',
        'coref.md.liberalChineseMD': 'false',

        # kbp
        # 'kbp.semgrex': 'edu/stanford/nlp/models/kbp/chinese/semgrex',
        # 'kbp.tokensregex': 'edu/stanford/nlp/models/kbp/chinese/tokensregex',
        # 'kbp.model': 'none',

        # entitylink
        'entitylink.wikidict': 'edu/stanford/nlp/models/kbp/wikidict_chinese.tsv.gz',

        'outputFormat': 'text'}
    coref1 = nlp.annotate(sentence, properties=pros)
    # print("命名实体识别：",nlp.ner(sentence))
    print("指代消解：", coref1)
    # print("句法分析：",nlp.parse(sentence))
    # print("语义依存分析：",nlp.dependency_parse(sentence))


if __name__ == '__main__':
    nlp = StanfordCoreNLP(local_corenlp_path, lang='zh', quiet=False)
    while True:
        sentence = input("请输入测试语句：")
        # vocab_tokenize(nlp,sentence)
        # vocab_pos(nlp, sentence)
        # vocab_ner(nlp, sentence)
        vocab_coref(nlp, sentence)
