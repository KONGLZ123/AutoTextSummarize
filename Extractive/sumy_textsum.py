from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.edmundson import EdmundsonSummarizer  # found this is the best as
import json
import re


LANGUAGE = "english"
SENTENCES_COUNT = 10
count = []
remove_index = [660, 778, 779, 781, 835, 839, 895, 896, 897, 898, 899, 901, 985, 989]

in_path = './bytecup2018/bytecup.corpus.validation_set.txt'
with open(in_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file.readlines()):
        out_path = './result/%d.txt' % (i + 1)
        out_file = open(out_path, 'w', encoding='utf-8')
        print(out_path)

        # 将str转换为dict
        data = json.loads(line.strip('\n'))
        article = data['content']

        if i + 1 in remove_index:
            result = article.split('{')
            out_file.write(result[0])
            print('result: ', result)
            continue

        dotlist = re.findall(r'[a-zA-Z]\.[a-zA-z]', article)
        print(dotlist)
        for sub in dotlist:
            tmp = sub.replace('.', '. ')
            article = article.replace(sub, tmp)
        # print(article)

        # article = re.sub('[a-zA-Z].[a-zA-z]', '. ', article)
        article = article.replace('\u2019', '\'')
        article = article.replace('\u201c', ' ')
        article = article.replace('\u00e9', ' ')
        article = article.replace('\u2014', ' ')
        article = article.replace('\u2013', ' ')
        article = article.replace('\u201c', ' ')
        article = article.replace('\u201d', ' ')
        article = article.replace('\u200b', ' ')
        article = article.replace('\u00c3', ' ')
        article = article.replace('\u0082', ' ')
        article = article.replace('\u00c2', ' ')
        article = article.replace('\u00a9', ' ')
        article = article.replace('\u00b0', ' ')
        article = article.replace('\u00ae', ' ')

        # article = article.replace('\"', '')
        # article = article.replace(r'\n', ' ')
        article = article.replace(r'\t', ' ')
        article = article.replace('#', ' ')
        article = article.replace('=', ' ')
        article = article.replace('&amp;', ' ')
        article = article.replace('READ ALSO:', ' ')
        article = article.replace('Ingredients:', ' ')

        # url = "https://en.wikipedia.org/wiki/Deep_learning"
        # parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
        # parser = PlaintextParser.from_file("./bytecup2018/pre_precesss_1000.txt", Tokenizer(LANGUAGE))
        parser = PlaintextParser.from_string(article, Tokenizer(LANGUAGE))

        # print("--LuhnSummarizer--")
        # summarizer = LuhnSummarizer(Stemmer(LANGUAGE))
        # summarizer.stop_words = get_stop_words(LANGUAGE)

        print("--LsaSummarizer--")
        summarizer = LsaSummarizer(Stemmer(LANGUAGE))
        summarizer.stop_words = ("I", "am", "the", "you", "are", "me", "is", "than", "that", "this",)

        # print("--EdmundsonSummarizer--")
        # summarizer = EdmundsonSummarizer(Stemmer(LANGUAGE))
        # words = ("deep", "learning", "neural")
        # summarizer.bonus_words = words
        # words = ("another", "and", "some", "next",)
        # summarizer.stigma_words = words
        # words = ("another", "and", "some", "next",)
        # summarizer.null_words = words

        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            print('abstract:', sentence)
            sentence = str(sentence)
            out_file.write(sentence.strip())

        out_file.close()

print(count)
