import os
import re
import struct
import collections
from nltk.tokenize import sent_tokenize
from tensorflow.core.example import example_pb2

counter = collections.Counter()
def convert_text2bin(docs, writer):
    global counter
    for i, fi in enumerate(docs):
        with open('cnn/stories/' + fi, 'r') as f:
            wholetext = f.read().lower()
            wholetext = re.sub(r'[^\x00-\x7F]+',  '', wholetext)
            wholetext = re.sub(r'(\s?[\']\s+[\']\s?)', " ' ", wholetext)
            wholetext = re.sub(r'(\s?[\"]\s+|\s+[\"]\s?)', ' " ', wholetext)
            wholetext = re.sub(r"(\'[s]\s+)", " 's ", wholetext)
            wholetext = wholetext.replace(".", " . ")
            wholetext = wholetext.replace(",", " , ")
            wholetext = wholetext.replace('-', ' - ')
            wholetext = wholetext.replace('?', ' ? ')
            wholetext = wholetext.replace('(', '( ')
            wholetext = wholetext.replace(')', ' )')
            data = wholetext.split("@highlight")
            news = data[0]
            highlights = data[1].replace('\n\n', '')
            news = (" ".join(news.split('\n\n'))).strip()
            sentences = sent_tokenize(news)
            news = '<d> <p> ' + ' '.join(['<s> ' + sentence + ' </s>' for sentence in sentences]) + ' </p> </d>'
            highlights = '<d> <p> <s> ' + highlights + ' </s> </p> </d>'
            words = (news + " " + highlights).split()
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
    print((float(len(docs)) / len(docs)) * 100, "%...." "converted\n\n")


files = os.listdir('cnn/stories/')
n_files = len(files)
print(n_files)

train = files[:int(n_files * 0.8)]
print(len(train))
validation=files[len(train):len(train)+int(n_files*0.12)]
print(len(validation))
test = files[len(train) + len(validation):]
print(len(test))

print("Generating Training Data\n")
with open('/home/kong/PycharmProjects/textsum/data/trainCNN.bin', 'wb') as writer:
    convert_text2bin(train, writer)
print("Generating Validation Data\n")
with open('/home/kong/PycharmProjects/textsum/data/validationCNN.bin', 'wb') as writer:
    convert_text2bin(validation, writer)
print("Generating Testing Data\n")
with open('/home/kong/PycharmProjects/textsum/data/testCNN.bin', 'wb') as writer:
    convert_text2bin(test, writer)


ntrain=len(train)
nval=len(validation)
ntest=len(test)
print("CNN Data Generated")
print("Train:\t\t",len(train))
print("Validation:\t",len(validation))
print("Test:\t\t",len(test))

temp=n_files

print("Generating Vocabulary")

mc=counter.most_common(200000-2)
with open("/home/kong/PycharmProjects/textsum/data/vocab", 'w') as writer:
    for word, count in mc:
        writer.write(word + ' ' + str(count) + '\n')
    writer.write('<UNK> 0\n')
    writer.write('<PAD> 0\n')


print("\n\nData Generation Finished...\n\n")

print("CNN Generated")
temp=0

print("Total Records", temp+n_files)
print("Total Train", ntrain+len(train))
print("Total Validation", nval+len(validation))
print("Total Test", ntest+len(test))
print("Vocab Generated with total no. of words:", len(mc)+2)



