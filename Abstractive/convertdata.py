import os
import re
import sys
import struct
import numpy as np
import collections
import json
from nltk.tokenize import sent_tokenize
from tensorflow.core.example import example_pb2

counter = collections.Counter()
temp=0
#train, test, validation split
tr_r=0.85
v_r=0.08

print('Generating data....')
files = []
for i in range(9):
    path = 'F:\\bytecup2018\\bytecup.corpus.train.%d.txt' % i
    in_file = open(path, 'r', encoding='utf-8', errors='ignore')
    files += in_file.readlines()
    in_file.close()

n_files=len(files)
print("Total Files:", n_files)

train=files[:int(n_files*0.8)]
validation=files[len(train):len(train)+int(n_files*0.12)]
test=files[len(train)+len(validation):]

def convert_text2bin(docs, writer):
    global counter
    for i, fi in enumerate(docs):
        item = json.loads(fi)
        title = item['title'].lower()
        wholetext = item['content'].lower()

        # wholetext=f.read().decode('utf8').lower()
        wholetext=re.sub(r'[^\x00-\x7F]+','', wholetext)
        wholetext=re.sub(r"(\s?[\']\s+|\s+[\']\s?)"," ' ", wholetext)
        wholetext=re.sub(r'(\s?[\"]\s+|\s+[\"]\s?)',' " ', wholetext)
        wholetext=re.sub(r"(\'[s]\s+)"," 's ", wholetext)
        wholetext=wholetext.replace('\n', ' ')
        wholetext=wholetext.replace("."," . ")
        wholetext=wholetext.replace(","," , ")
        wholetext=wholetext.replace('-',' - ')
        wholetext=wholetext.replace('?',' ? ')
        wholetext=wholetext.replace('(','( ')
        wholetext=wholetext.replace(')',' )')

        # data=wholetext.split("@highlight")

        news=wholetext
        highlights=title
        print('news:' + news + '\n' + 'highlights: ' + highlights)
        # news=(" ".join(news.split('\n\n'))).strip()
        sentences = sent_tokenize(news)
        news = '<d> <p> ' + ' '.join(['<s> ' + sentence + ' </s>' for sentence in sentences]) + ' </p> </d>'
        highlights = '<d> <p> <s> ' + highlights + ' </s> </p> </d>'
        words = (news+" "+highlights).split()
        print('words', words)
        counter.update(words)

        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([(' '.join(news.split())).encode('utf-8')])
        tf_example.features.feature['abstract'].bytes_list.value.extend([(' '.join(highlights.split())).encode('utf-8')])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
        if i%3000 == 0:
            print(int((float(i)/ len(docs))*100), "%")
    print((float(len(docs))/ len(docs))*100, "%...." "converted\n\n")

print("Generating Training Data\n")
with open('F:\\bytecup2018\\data\\train.bin', 'wb') as writer:
    convert_text2bin(train,writer)
print("Generating Validation Data\n")
with open('F:\\bytecup2018\\data\\validation.bin', 'wb') as writer:
    convert_text2bin(validation,writer)
print("Generating Testing Data\n")
with open('F:\\bytecup2018\\data\\test.bin', 'wb') as writer:
    convert_text2bin(test,writer)

ntrain=len(train)
nval=len(validation)
ntest=len(test)
print("Data Generated")
print("Train:\t\t",len(train))
print("Validation:\t",len(validation))
print("Test:\t\t",len(test))

temp=n_files

print("Generating Vocabulary")

mc=counter.most_common(200000-2)
with open("F:\\bytecup2018\\data\\vocab", 'w', encoding='utf-8') as writer:
    for word, count in mc:
        writer.write(word + ' ' + str(count) + '\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')


print("\n\nData Generation Finished...\n\n")

print("Generated")
temp=0

print("Total Records",temp+n_files)
print("Total Train",ntrain+len(train))
print("Total Validation",nval+len(validation))
print("Total Test",ntest+len(test))
print("Vocab Generated with total no. of words:",len(mc)+2)