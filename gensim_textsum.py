# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s %(message)s', level=logging.INFO)

import re
import json
import requests
from gensim.summarization import summarize
from gensim.summarization import keywords

# text = "Thomas A. Anderson is a man living two lives. By day he is an " + \
#     "average computer programmer and by night a hacker known as " + \
#     "Neo. Neo has always questioned his reality, but the truth is " + \
#     "far beyond his imagination. Neo finds himself targeted by the " + \
#     "police when he is contacted by Morpheus, a legendary computer " + \
#     "hacker branded a terrorist by the government. Morpheus awakens " + \
#     "Neo to the real world, a ravaged wasteland where most of " + \
#     "humanity have been captured by a race of machines that live " + \
#     "off of the humans' body heat and electrochemical energy and " + \
#     "who imprison their minds within an artificial reality known as " + \
#     "the Matrix. As a rebel against the machines, Neo must return to " + \
#     "the Matrix and confront the agents: super-powerful computer " + \
#     "programs devoted to snuffing out Neo and the entire human " + \
#     "rebellion. "

#text = requests.get('http://rare-technologies.com/the_matrix_synopsis.txt')
# text = 'When I look into your eyes.It\u2019s like watching the night sky.Or a beautiful sunriseWell, there\u2019s so much they holdAnd just like them old stars.I see that you\u2019ve come so far.To be right where you are.How old is your soul?Well, I won\u2019t give up on usEven if the skies get roughI\u2019m giving you all my loveI\u2019m still looking upAnd when you\u2019re needing your spaceTo do some navigatingI\u2019ll be here patiently waitingTo see what you find\u2018Cause even the stars they burnSome even fall to the earthWe\u2019ve got a lot to learnGod knows we\u2019re worth itNo, I won\u2019t give upI don\u2019t wanna be someone who walks away so easilyI\u2019m here to stay and make the difference that I can makeOur differences they do a lot to teach us how to useThe tools and gifts we got, yeah, we got a lot at stakeAnd in the end, you\u2019re still my friend at least we did intendFor us to work we didn\u2019t break, we didn\u2019t burnWe had to learn how to bend without the world caving inI had to learn what I\u2019ve got, and what I\u2019m not, and who I amI won\u2019t give up on usEven if the skies get roughI\u2019m giving you all my loveI\u2019m still looking up, I\u2019m still looking up.Well, I won\u2019t give up on us (no I\u2019m not giving up)God knows I\u2019m tough enough (I am tough, I am loved)We\u2019ve got a lot to learn (we\u2019re alive, we are loved)God knows we\u2019re worth it (and we\u2019re worth it)I won\u2019t give up on usEven if the skies get roughI\u2019m giving you all my loveI\u2019m still looking up'
#
# print(text)
#
# sum = summarize(text)
# print(sum)
#
# keywords = keywords(text)
# print(keywords)
#
# sum = summarize(text, word_count=25)
# print(sum)

# text = 'University.The, hello. world.haha'
# dotlist = re.findall(r'[a-zA-Z]\.[a-zA-z]', text)
# print(dotlist)
# for sub in dotlist:
#     tmp = sub.replace('.', '. ')
#     text = text.replace(sub, tmp)
# print(text)

count = []
remove_index = [778, 779, 781, 835, 839, 895, 896, 897, 898, 899, 901, 985, 989]

in_path = './bytecup2018/bytecup.corpus.validation_set.txt'
pre_file = open('./bytecup2018/pre_precesss_1000.txt', 'w', encoding='utf-8')
with open(in_path, 'r', encoding='utf-8') as file:
    try:
        for i, line in enumerate(file.readlines()):
            out_path = './bytecup2018/result/%d.txt' % (i + 1)
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
            # sentence = parser.removePunctations(sentence)
            # sentence = parser.removeStopWords(list(sentence.split(' ')))

            pre_file.write(article + '\n')
            sum = summarize(article, word_count=150)
            if len(sum) < 1:
                count.append(i + 1)
            print(i + 1, sum)

            out_file.write(sum)

            out_file.close()
    finally:
        file.close()
out_file.close()

print(count)