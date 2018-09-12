from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from string import punctuation
from heapq import nlargest
# import urllib2
from bs4 import BeautifulSoup
import json
import re

class FrequencySummarizer:
  def __init__(self, min_cut=0.1, max_cut=0.9):
    """
     Initilize the text summarizer.
     Words that have a frequency term lower than min_cut
     or higer than max_cut will be ignored.
    """
    self._min_cut = min_cut
    self._max_cut = max_cut
    self._stopwords = set(stopwords.words('english') + list(punctuation))

  def _compute_frequencies(self, word_sent):
    """
      Compute the frequency of each of word.
      Input:
       word_sent, a list of sentences already tokenized.
      Output:
       freq, a dictionary where freq[w] is the frequency of w.
    """
    freq = defaultdict(int)
    for s in word_sent:
      for word in s:
        if word not in self._stopwords:
          freq[word] += 1
    # frequencies normalization and fitering
    m = float(max(freq.values()))
    for w in list(freq.keys()):
      freq[w] = freq[w]/m
      if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
        del freq[w]
    return freq

  def summarize(self, text, n):
    """
      Return a list of n sentences
      which represent the summary of text.
    """
    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = [word_tokenize(s.lower()) for s in sents]
    self._freq = self._compute_frequencies(word_sent)
    ranking = defaultdict(int)
    for i,sent in enumerate(word_sent):
      for w in sent:
        if w in self._freq:
          ranking[i] += self._freq[w]
    sents_idx = self._rank(ranking, n)
    return [sents[j] for j in sents_idx]

  def _rank(self, ranking, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, ranking, key=ranking.get)

def get_only_text(url):
 """
  return the title and the text of the article
  at the specified url
 """
 page = urllib2.urlopen(url).read().decode('utf8')
 soup = BeautifulSoup(page)
 text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
 return soup.title.text, text


# feed_xml = urllib2.urlopen('http://feeds.bbci.co.uk/news/rss.xml').read()
# feed = BeautifulSoup(feed_xml.decode('utf8'))
# to_summarize = map(lambda p: p.text, feed.find_all('guid'))
#
# fs = FrequencySummarizer()
# for article_url in to_summarize[:5]:
#   title, text = get_only_text(article_url)
#   print('----------------------------------')
#   print(title)
#   for s in fs.summarize(text, 2):
#    print('*',s)


fs = FrequencySummarizer()
remove_index = [778, 779, 781, 835, 839, 895, 896, 897, 898, 899, 901, 985, 989]

count = []
in_path = './bytecup2018/bytecup.corpus.validation_set.txt'
with open(in_path, 'r', encoding='utf-8') as file:
    try:
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

            sent = article.strip()
            sum = fs.summarize(str(sent), 2)
            if len(sum) < 1:
                count.append(i + 1)
            print(i + 1, sum)

            out_file.write(sum[0])
            out_file.close()
    finally:
        file.close()
out_file.close()

print(count)