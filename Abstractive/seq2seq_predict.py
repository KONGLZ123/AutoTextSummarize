#! -*- coding: utf-8 -*-

import numpy as np
import os, json
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from nltk.tokenize import RegexpTokenizer
from nltk.parse.corenlp import CoreNLPParser
import h5py

min_count = 1
maxlen = 400
batch_size = 32
epochs = 50
char_size = 128
heads = []
desc = []

# perpare data
maxLines = 1000
i = 0
chars = {}

# with open('../data/byte_train_correct_0.txt', 'r', encoding='utf-8', errors='ignore') as infile:
with open('/home/kong/PycharmProjects/bytecup2018/bytecup2018/bytecup.corpus.train.0.txt', 'r', encoding='utf-8', errors='ignore') as infile:
    for line in infile.readlines():
        # item = line.split('   ')
        # title = item[0].split(':')[1].replace('\n', ' ').lower()
        # content = item[1].split(':')[1].replace('\n', ' ').lower()

        lineParsed = json.loads(line.strip('\n'))
        content = lineParsed['content'].strip().replace('\n', ' ').lower()
        title = lineParsed['title'].strip().replace('\n', ' ').lower()
        print('title:', title)
        print('content:', content)

        # nltk 分词
        tokenizer = RegexpTokenizer(r'\w+')
        content = tokenizer.tokenize(content)
        title = tokenizer.tokenize(title)

        # standford core nlp tokenize
        # st = CoreNLPParser()
        # title = list(st.tokenize(title))
        # content = list(st.tokenize(content))
        print('remove punctuation:', content)

        for w in content:
            chars[w] = chars.get(w, 0) + 1
        for w in title:
            chars[w] = chars.get(w, 0) + 1

        i += 1
        if (i >= maxLines):
            break

chars = {i: j for i, j in chars.items() if j >= min_count}
# 0: mask
# 1: unk
# 2: start
# 3: end
id2char = {i + 4: j for i, j in enumerate(chars)}
char2id = {j: i for i, j in id2char.items()}
json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [2] + ids + [3]
    else: # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ' '.join([id2char.get(i, ' ') for i in ids])


def padding(x):
    # padding至batch内的最大长度
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


def data_generator():
    # 数据生成器
    X, Y = [], []
    global maxLines
    print('maxLines:', maxLines)
    i = 0
    while True:
        # infile = open('../data/byte_train_correct_0.txt', 'r', encoding='utf-8', errors='ignore')
        infile = open('/home/kong/PycharmProjects/bytecup2018/bytecup2018/bytecup.corpus.train.0.txt', 'r',
                      encoding='utf-8', errors='ignore')
        for line in infile.readlines():
            # item = line.split('   ')
            # title = item[0].split(':')[1].replace('\n', ' ').lower()
            # content = item[1].split(':')[1].replace('\n', ' ').lower()

            lineParsed = json.loads(line.strip('\n'))
            content = lineParsed['content'].strip().replace('\n', ' ').lower()
            title = lineParsed['title'].strip().replace('\n', ' ').lower()

            # nltk 分词
            tokenizer = RegexpTokenizer(r'\w+')
            content = tokenizer.tokenize(content)
            title = tokenizer.tokenize(title)

            # standford core nlp tokenize
            # st = CoreNLPParser()
            # title = list(st.tokenize(title))
            # content = list(st.tokenize(content))

            X.append(str2id(content))
            Y.append(str2id(title, start_end=True))

            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X, Y], None
                X, Y = [], []

            i = i + 1
            if (i >= maxLines):
                break
        infile.close()


# 搭建seq2seq模型

x_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x = x_in
y = y_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(y)


def to_one_hot(x): # 输出一个词表大小的向量，来标记该词是否在文章出现过
    x, x_mask = x
    x = K.cast(x, 'int32')
    x = K.one_hot(x, len(chars)+4)
    x = K.sum(x_mask * x, 1, keepdims=True)
    x = K.cast(K.greater(x, 0.5), 'float32')
    return x


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def build(self, input_shape):
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],)
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs):
        x_outs = K.exp(self.log_scale) * inputs + self.shift
        return x_outs


x_one_hot = Lambda(to_one_hot)([x, x_mask])
x_prior = ScaleShift()(x_one_hot) # 学习输出的先验分布（标题的字词很可能在文章出现过）

embedding = Embedding(len(chars)+4, char_size)
x = embedding(x)
y = embedding(y)

# encoder，双层双向LSTM
# x = Bidirectional(CuDNNLSTM(int(char_size/2), return_sequences=True))(x)
# x = Bidirectional(CuDNNLSTM(int(char_size/2), return_sequences=True))(x)
x = Bidirectional(LSTM(int(char_size//2), return_sequences=True))(x)
x = Bidirectional(LSTM(int(char_size//2), return_sequences=True))(x)

# decoder，双层单向LSTM
# y = CuDNNLSTM(char_size, return_sequences=True)(y)
# y = CuDNNLSTM(char_size, return_sequences=True)(y)
y = LSTM(char_size, return_sequences=True)(y)
y = LSTM(char_size, return_sequences=True)(y)


class Interact(Layer):
    """交互层，负责融合encoder和decoder的信息
    """
    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)
    def build(self, input_shape):
        in_dim = input_shape[0][-1]
        out_dim = input_shape[1][-1]
        print('in_dim:', in_dim)
        print('out_dim:', out_dim)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')
    def call(self, inputs):
        q, v, v_mask = inputs
        k = v
        mv = K.max(v - (1. - v_mask) * 1e10, axis=1, keepdims=True)  # maxpooling1d
        mv = mv + K.zeros_like(q[:, :, :1])  # 将mv重复至“q的timesteps”份
        # 下面几步只是实现了一个乘性attention
        qw = K.dot(q, self.kernel)
        a = K.batch_dot(qw, k, [2, 2]) / 10.
        a -= (1. - K.permute_dimensions(v_mask, [0, 2, 1])) * 1e10
        a = K.softmax(a)
        o = K.batch_dot(a, v, [2, 1])
        # 将各步结果拼接
        return K.concatenate([o, q, mv], 2)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2] + input_shape[1][2]*2)


xy = Interact()([y, x, x_mask])
xy = Dense(512, activation='relu')(xy)
xy = Dense(len(chars)+4)(xy)
xy = Lambda(lambda x: (x[0]+x[1])/2)([xy, x_prior])  # 与先验结果平均
xy = Activation('softmax')(xy)

# 交叉熵作为loss，但mask掉padding部分
cross_entropy = K.sparse_categorical_crossentropy(y_in[:, 1:], xy[:, :-1])
loss = K.sum(cross_entropy * y_mask[:, 1:, 0]) / K.sum(y_mask[:, 1:, 0])

model = Model([x_in, y_in], xy)
model.add_loss(loss)
model.compile(optimizer=Adam(1e-3))

model.load_weights('/home/kong/PycharmProjects/AutoTextSummarize/Abstractive/model1.weights')

def gen_title(s, topk=3):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(s)] * topk) # 输入转id
    yid = np.array([[2]] * topk) # 解码均以<start>开通，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    for i in range(50): # 强制要求标题不超过50字
        proba = model.predict([xid, yid])[:, i, 3:]  # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _yid = []  # 暂存的候选目标序列
        _scores = []  # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]+3])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]+3])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == 3: # 找到<end>就返回
                return id2str(_yid[k])
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果50字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


# s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
# s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
# s1 = u"When someone commits a murder they typically go to extreme lengths to cover up their brutal crime . The harsh prison sentences that go along with killing someone are enough to deter most people from ever wanting to be caught, not to mention the intense social scrutiny they would face . Occasionally, however, there are folks who come forward and admit guilt in their crime . This can be for any number of reasons, like to gain notoriety or to clear their conscience, though, in other instances, people do it to come clean to the people they care about . </s> <s> When Rachel Hutson was just 19 years old, she murdered her own mother in cold blood . </s> <s> As heinous and unimaginable as her crime was, it was what she did after that shocked people the most…  Rachel was just a teenager when she committed an unthinkable act against her own other… While that in and of itself was a heinous crime, it’s what Rachel did in the aftermath of her own mother’s murder that shook people to their core . You’re not going to believe what strange thing she decided to do next… It’s hard to understand what drove Rachel to commit this terrible act, but sending the photo afterward seems to make even less sense . Share this heartbreaking story with your friends below ."
# s2 = u"James Milner may not be one of the most exciting players in world football but he is one of the most effective, as underlined by the fact that the Liverpool ace is on the verge of making Champions League history . Ahead of his side's semi-final first-leg meeting with Roma at Anfield, the England international needs just one more assist to break the tournament's all-time record for a single season . Indeed, Milner has already created eight goals during the Reds' remarkable run to the last four – three more than his nearest rivals, team-mate Roberto Firmino and Luis Suarez of the already-eliminated Barcelona . Should the 32-year-old midfielder turn provider again against Roma, he will claim outright possession of the record from Neymar and Wayne Rooney, both of whom racked up seven assists, in 2016-17 and 2013-14, respectively . Milner's achievement is made all the more remarkable for the fact that he has played only 611 minutes – fewer than both Neymar (797) and Rooney (765) . He actually started Liverpool's Champions League campaign on the bench, failing to see any game time in the draws with Sevilla and Spartak Moscow . However, as soon as he was added to Jurgen Klopp's starting line-up, the Reds began winning, with Milner racking up an assist in both the away and home wins over Maribor . Milner set up three goals in the 7-0 demolition of Spartak, which saw Liverpool progress to the knockout stage as winners of Group E . Milner teed Firmino up for Liverpool's fourth in their 5-0 win at Porto in the last 16 before doing likewise for Alex Oxlade-Chamberlain in the rousing 3-0 victory over Manchester at Anfield in the quarter-finals . As a result, the Reds' unlikely hero is now poised to do something no player has ever done before in Champions League history! ."

s1 = u"Being the daughter of Hollywood superstar Tom Cruise and America's sweetheart Katie Holmes, people have always been interested in Suri Cruise .Despite her parents' shocking split in 2012, the young child appears to grow up to be a happy child even with the absence of her father . In fact, the mother-daughter tandem recently enjoyed a nice dinner date together that shows off their incredibly close relationship .   Daily Mail has provided a couple of shots of Katie Holmes and Suri Cruise strolling the chilly streets of New York City on Monday to head to their dinner destination . Once inside the restaurant, the 39-year-old actress was snapped looking lovingly at her daughter as she enjoyed her meal . After Katie Holmes ended her 6-year marriage to Tom Cruise, the actress made sure that Suri Cruise will not feel that something is missing in her life . While some might see her split from the actor as an unfortunate matter, the Dawson's Creek star takes this as an opportunity to be the best parent that she can be for her daughter . While Katie Holmes is enjoying every moment that she spends with Suri Cruise, Tom Cruise remains absent in the 11-year-old's life . Suri used to be daddy's little girl, but after the actress filed for divorce in 2012, the Mission: Impossible 6 actor's relationship with his daughter quickly went downhill . Although the All We Had star never talked about the real reason why she ended their marriage, it was believed that the 55-year-old actor's ties with Scientology may have caused the split . Being the face of Scientology for years and a proud loyalist, Tom Cruise full-heartedly follows the rules to a tee . While his actions and loyalty are highly-praised by the controversial church's followers and leader, David Miscavige, Katie Holmes and people outside the cult-like religion just can't seem to fathom as to how he could put Scientology before his daughter . Despite this unfortunate situation, it appears that Katie Holmes and Suri Cruise have already gotten used to their new norm . In fact, the actress is already enjoying a new romance with Jamie Foxx, so Tom Cruise is clearly out of the picture . As for Suri Cruise, an earlier report revealed that she has “stopped missing” her dad . Suri Cruise 2018: Katie Holmes Bonds With Daughter During Dinner Date While Tom Cruise Still MIA is an article from: The Inquisitr ."
s2 = u"By Disha RawalIndia's multinational giant Oil and Natural Gas Corporation Limited has bought the government's 51% stake in Hindustan Petroleum Corporation Limited, making it the majority stakeholder . This merger creates a new energy behemoth, with a presence in every part of the energy market . This Rs 36,915 crore deal is not just any agreement but is a precedent-setting macroeconomic event . This is because, with this deal, the Government of India has exceeded its disinvestment target for the year, which was pegged at Rs . 72,500 crores, which may greatly help the government tackle its fiscal deficit . The rise of an energy behemothThe process toward this deal began on the 19th of July 2017, when the cabinet gave its approval-in-principle . The Finance Ministry then initiated the process of determining the price and other conditions . The central government sought to integrate different assets in the new private company in order to establish create a new major player in the industry which has sufficient breadth to bear increased risk and to attract greater investment . This is good news for India's oil and gas sector, as it is estimated that nearly 300 billion is needed in investments by 2040 to develop this industry and meet the growing demand for petrol products in the country . The money raised from the government's disinvestment in the industry is also expected to help it meet its deficit target of 3.2% of GDP . Lately, this target was looking difficult to achieve, given the falling revenues collected through the Goods and Services Tax (GST) . Accordingly, the government has also had cut its borrowing target by a whopping 60%, to around Rs, 20,000 crores . Already, apart from this latest deal, disinvestment proceeds to the government in the current year have contained what could have otherwise been a difficult fiscal situation for the nation . The government reduced its stake in Public Service Units (PSUs) like OIL, NTPC, and HUDCO . In the Union Budget presented on the first of February last year, the Finance Minister set a disinvestment target of Rs 72,500 crore, including Rs 46,500 crore from the disinvestment of central public sector enterprises, Rs 15,000 crore from strategic sales and Rs 11,000 crore from listing public insurance companies . This would be the first time that a government's disinvestment target is met . What to watch out for in the budgetIt has been suggested that the government may reduce its disinvestment targets for the next fiscal year . This deal, however, has given a lease on life to the disinvestment program . This may also help the national economy, as PSUs have had a long history of posting negative rates of return on the capital that the state invests in them . However, it remains to be seen how this deal will be translated into policy action . Featured Image Source: Wikimedia ."
# abstract=Suri Cruise 2018: Katie Holmes Bonds With Daughter During Dinner Date While Tom Cruise Still MIA
# abstract=How the ONGC-HPCL deal may help grow India's economy

# nltk 分词
tokenizer = RegexpTokenizer(r'\w+')
s1 = tokenizer.tokenize(s1)
s2 = tokenizer.tokenize(s2)

print('s1:', gen_title(s1))
print('s2:', gen_title(s2))


