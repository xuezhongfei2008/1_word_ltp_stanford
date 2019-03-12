# -*- coding: utf-8 -*-
import sys
import os
import sys
import nltk
from nltk.tree import Tree  # 导入nltk tree结构
from nltk.grammar import DependencyGrammar  # 导入依存句法包
from nltk.parse import *
from pyltp import *  # 导入ltp应用包
import re

file_path='/opt/algor/gongxf/software/LTP/ltp_data_v3.4.0/'             ####linux路径
# file_path='D:\\LTP\ltp_data_v3.4.0\\'                           #windows路径

#切词
def ltp_cut():
    sent = "您好，分期付款您可以在下单结算界面选择“信用卡分期”，支付成功后，" \
           "银行会一次性扣除总金额，后期每个月给银行还款分期的金额；或者您选择任性付分期，" \
           "3/6/9/12期都是可以的哦（部分商品会有免息活动），具体以网页信息为准，" \
           "暂不支持银行卡分期。门店分期具体咨询门店哦。"
    model_path = file_path+"cws.model"
    segmentor = Segmentor()
    segmentor.load(model_path)
    words = segmentor.segment(sent)
    print(" | ".join(words))
    #上述分词粒度过细，为了获得更精确的结果可以将错分的结果合并为专有名词。这就是分词结果的后处理过程，即一般外部用户词典的构成原理
    postdict = {"解 | 空间": "解空间",
                "深度 | 优先": "深度优先"}

    seg_sent = " | ".join(words)
    for key in postdict:
        seg_sent = seg_sent.replace(key, postdict[key])
    print(seg_sent)

    #加入用户自定义字典
    user_dict = file_path+"finWordDict.txt"  # 外部专有名词词典
    segmentor1 = Segmentor()
    segmentor1.load_with_lexicon(model_path, user_dict)  # 加载专有名词词典
    words = segmentor.segment(sent)
    print(" | ".join(words))

#词性标注
def ltp_pos():
    sent = "在 包含 问题 的 所有 解 的 解空间树 中 ， 按照 深度优先 搜索 的 策略 ， 从 根节点 出发 深度 探索 解空间树 。"
    words = sent.split(" ")
    postagger = Postagger()  # 实例化词性标注类
    postagger.load(file_path+'pos.model')
    postags = postagger.postag(words)
    for word, postag in zip(words, postags):
        print(word + "/" + postag)

#命名实体识别
def ltp_ner():
    sent = "欧洲 东部 的 罗马尼亚 ， 首都 是 布加勒斯特 ， 也 是 一 座 世界性 的 城市 。"
    words = sent.split(" ")
    postagger = Postagger()
    postagger.load(file_path+"pos.model")  # 导入词性标注模块
    postags = postagger.postag(words)
    recognizer = NamedEntityRecognizer()
    recognizer.load(file_path+"ner.model")  # 导入命名实体识别模块
    netags = recognizer.recognize(words, postags)
    for word, postag, netag in zip(words, postags, netags):
        print(word + "/" + postag + "/" + netag)

#句法分析
def ltp_parser():
    words = "谢霆锋 的 爸爸 是 谢贤 。".split(" ")  # 例句
    print(words)
    postagger = Postagger()  # 词性标注
    postagger.load(file_path+"pos.model")
    postags = postagger.postag(words)
    print("len(postags)",len(postags))

    parser = Parser()  # 句法解析
    parser.load(file_path+"parser.model")
    arcs = parser.parse(words, postags)
    arclen = len(arcs)
    print("arclen",arclen)
    conll = ""
    for i in range(arclen):  # 构建Conll标准的数据结构
        if arcs[i].head == 0:
            arcs[i].relation = "ROOT"
        conll += "\t" + words[i] + "(" + postags[i] + ")" + "\t" + postags[i] + "\t" + str(arcs[i].head) + "\t" + arcs[i].relation + "\n"
    print("conll",conll)
    conlltree = DependencyGraph(conll)  # 转换为依存句法图
    tree = conlltree.tree()  # 构建树结构
    # tree.draw()

def ltp_semantic_annotation():
    MODELDIR = file_path
    sentence = "欧洲东部的罗马尼亚，首都是布加勒斯特，也是一座世界性的城市。"

    segmentor = Segmentor()
    segmentor.load(os.path.join(MODELDIR, "cws.model"))
    words = segmentor.segment(sentence)
    wordlist = list(words)  # 从生成器变为列表元素
    print("分词：",wordlist)


    postagger = Postagger()
    postagger.load(os.path.join(MODELDIR, "pos.model"))
    postags = postagger.postag(words)
    print("词性标注：",postags)

    parser = Parser()
    parser.load(os.path.join(MODELDIR, "parser.model"))
    arcs = parser.parse(words, postags)
    print("句法解析：",arcs)

    recognizer = NamedEntityRecognizer()
    recognizer.load(os.path.join(MODELDIR, "ner.model"))
    netags = recognizer.recognize(words, postags)
    print("命名实体识别：",netags)

    # 语义角色标注
    labeller = SementicRoleLabeller()
    labeller.load(os.path.join(MODELDIR, "pisrl_win/"))
    roles = labeller.label(words, postags, netags, arcs)
    print("语义标注：",roles)

    # 输出标注结果
    for role in roles:
        print('rel:', wordlist[role.index])  # 谓词
        for arg in role.arguments:
            if arg.range.start != arg.range.end:
                print(arg.name, ' '.join(wordlist[arg.range.start:arg.range.end]))
            else:
                print(arg.name, wordlist[arg.range.start])

if __name__=="__main__":
    ltp_cut()
    # ltp_pos()
    # ltp_ner()
    # ltp_parser()
    # ltp_semantic_annotation()
    # pass