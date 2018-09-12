import json

# def summarize(title, text, category="Undefined", source="Undefined", count=5):
#     tt = Summarizer()
#     result = tt.summarize(text, title, source, category)
#     result = tt.sortSentences(result[:count])
#     result = [res['sentence'] for res in result]
#     return result

# title = "Limitations of the GET method in HTTP"
# text = "We spend a lot of time thinking about web API design, and we learn a lot from other APIs and discussion with their authors. In the hopes that it helps others, we want to share some thoughts of our own. In this post, we’ll discuss the limitations of the HTTP GET method and what we decided to do about it in our own API.  As a rule, HTTP GET requests should not modify server state. This rule is useful because it lets intermediaries infer something about the request just by looking at the HTTP method.  For example, a browser doesn’t know exactly what a particular HTML form does, but if the form is submitted via HTTP GET, the browser knows it’s safe to automatically retry the submission if there’s a network error. For forms that use HTTP POST, it may not be safe to retry so the browser asks the user for confirmation first.  HTTP-based APIs take advantage of this by using GET for API calls that don’t modify server state. So if an app makes an API call using GET and the network request fails, the app’s HTTP client library might decide to retry the request. The library doesn’t need to understand the specifics of the API call.  The Dropbox API tries to use GET for calls that don’t modify server state, but unfortunately this isn’t always possible. GET requests don’t have a request body, so all parameters must appear in the URL or in a header. While the HTTP standard doesn’t define a limit for how long URLs or headers can be, most HTTP clients and servers have a practical limit somewhere between 2 kB and 8 kB.  This is rarely a problem, but we ran up against this constraint when creating the /delta API call. Though it doesn’t modify server state, its parameters are sometimes too long to fit in the URL or an HTTP header. The problem is that, in HTTP, the property of modifying server state is coupled with the property of having a request body.  We could have somehow contorted /delta to mesh better with the HTTP worldview, but there are other things to consider when designing an API, like performance, simplicity, and developer ergonomics. In the end, we decided the benefits of making /delta more HTTP-like weren’t worth the costs and just switched it to HTTP POST.  HTTP was developed for a specific hierarchical document storage and retrieval use case, so it’s no surprise that it doesn’t fit every API perfectly. Maybe we shouldn’t let HTTP’s restrictions influence our API design too much.  For example, independent of HTTP, we can have each API function define whether it modifies server state. Then, our server can accept GET requests for API functions that don’t modify server state and don’t have large parameters, but still accept POST requests to handle the general case. This way, we’re opportunistically taking advantage of HTTP without tying ourselves to it."
#
# sentence = summarize(title, text)
# print(sentence)


# parser = Parser()
# with open('./bytecup2018/bytecup.corpus.validation_set.txt', 'r', encoding='utf-8') as file:
#     try:
#         for i, line in enumerate(file.readlines()):
#             # 将str转换为dict
#             data = json.loads(line.strip('\n'))
#             sentence = data['content']
#             print('raw:' + sentence)
#             # sentence = summarize(' ', data['content'])
#             # sentence = parser.removePunctations(sentence)
#             # sentence = parser.removeStopWords(list(sentence.split(' ')))
#             path = './bytecup2018/result/{}.txt'.format(i+1)
#             print(path)
#             output = open(path, 'w', encoding='utf-8')
#             sentence = summarize(sentence, ratio=0.1, words=20)
#             output.write(sentence)
#             output.close()
#             print('summarzie:' + sentence)
#             print('len:', len(sentence))
#     finally:
#         file.close()

# abstract=b"b'<d> <p> <s> mexican financial markets end turbulent week . </s> </p> </d>'"
# article=b'b"<d> <p> <s> the mexican peso appeared to stabilize friday at the end of a turbulent week on financial markets that saw a record low for the currency amid a wave of rumors and signs of growing unease about the economy . </s> <s> the currency closed here at #.## pesos to the dollar , unchanged from thursday but down some ## centavos from a week earlier . </s> <s> earlier in the week the mexican currency slid to as low as #.## to the dollar in interbank trading before rallying , helped by central bank intervention . </s> <s> president ernesto zedillo sought to assure investors , telling a gathering of businessmen in acapulco that `` the episodes of financial instability are past and they will have to end . </s> <s> \'\' `` the mexican economy today has safe and sound foundations to push growth with the creation of jobs , \'\' zedillo added . </s> <s> financial sources said the intervention on the markets friday to stabilize the peso was `` modest , \'\' and similar to thursday \'s . </s> <s> late thursday , the bank acknowledged , `` we intervened ... on the currency markets , as is done in germany , japan or the united states , because speculation had pushed exchange rates to irrational levels . </s> <s> \'\' stocks survived the week relatively unscathed , with the bmw index closing friday at #,###.## , a rise of #.## points from a week ago . </s> <s> the markets were shaken by a series of rumors ranging from a shakeup in zedillo \'s cabinet to a military coup . </s> <s> while the rumors proved unfounded , a strong demand for dollars forced the government to boost interest rates on short-term treasury notes -lrb- cetes -rrb- to ##.## percent . </s> <s> that move backfired , rattling the markets even further . </s> <s> analysts said the markets remained uneasy about the government \'s ability to deal with the financial crisis that began with a plunge in the peso nearly a year ago . </s> <s> some predicted the government would have to come up with new measures to reassure investors as early as next week . </s> <s> in a report released friday , the government said the september trade surplus was ### million dollars , bringing the #### surplus to #.# billion dollars . </s> <s> that compares with a deficit of ##.# billion dollars in the same period in #### . </s> <s> finance minister guillermo ortiz told lawmakers earlier this week that strong exports would bring the economy back to positive growth in #### , predicting a rise of #.# to two percent in the economy next year . . </s> </p> </d>"'
# publisher=b"b'AFP'"
#
#
# publisher=b"b'AFP'"
# abstract=b"b'<d> <p> <s> opposition activists paralyse calcutta . </s> </p> </d>'"
# article=b'b"<d> <p> <s> activists from india \'s ruling congress -lrb- i -rrb- party blocked traffic and held up trains wednesday in the marxist-ruled state of west bengal to protest at alleged deaths of party workers in police custody . </s> <s> shops , offices and schools were shut in the state capital calcutta and other towns as congress protested to press for action against officers deemed responsible . </s> <s> protesters set up roadblocks and squatted on rail tracks stranding tens of thousands of commuters who daily travel to work in calcutta , witnesses said . </s> <s> mamta banerjee , leader of congress \'s west bengal chapter , said the day-long general strike was over the deaths in custody of a total of ### party activists during the ##-year rule of marxists . </s> <s> congress is in the opposition in the state . </s> <s> `` we demand immediate action . </s> <s> we also demand compensation for those murdered in police custody , \'\' the congress leader said , threatening to launch an indefinite fast if her demands were not met . </s> <s> west bengal chief minister jyoti basu has dismissed the congress claims as a move to discredit his communist party of india -lrb- marxist -rrb- government in the eastern state . . </s> </p> </d>"'
#
# {"content": "The whole yearly game comes to the last one, and we are going to practice how to proper ending in this month missions.First of all, we have to inventory our game map, take a look at what has been completed?Have you unlocked some of your hidden powers? For example, you never knew you could do one thing in succession for more than thirty days? End up you do it for six months?Or you stepped out of your comfort zone, originally thought very uncomfortable, but eventually bring the joy of growth?These are all things that need to be actually recorded.Through the recorded actions, we re-examine ourselves and give ourselves new self-worth and self-interpretation.Celebrate your success, no matter how much you have done, you have made some different attempts.Finally, make a simple assessment report for this year\u2019s game, write down all the remembered progress, summed up your greatest achievement, and the biggest failure.There are many free resources online that allow us to review the growth of this year. You can make your own annual assessment report based on thirty questions from bohoberry\u2019s December challenge.Happy transgame 2017, please take the inspiration and have a nice 2018.",
#  "id": 98,
#  "title": "Happy transgame 2017, please take the inspiration and have a nice 2018."}

# sentence = "abstract=<d> <p> <s> Coca Cola Sets New 2030 Recycling Goals To Cut Packaging Waste </s> </p> </d>	article=<d> <p> <s> The Coca-Cola Company has just announced its new approach to significantly reducing waste . </s> <s> The company's vision for a World Without Waste involves recycling and changing its approach to packaging . </s> <s> Investing In The Planet For A World Without WasteOn Jan . </s> <s> 19, the Coca-Cola Company shared its global vision for a waste-free world . </s> <s> The company's press release states that the vision begins by understanding that food and beverage packaging are important parts of a modern world, but that there are so many things that can be done to reduce the waste . </s> <s> As such, the company has set two key goals, which it aims to have achieved by the year 2030 . </s> <s> First, for every bottle or can that Coca-Cola sells, they intend to take another one to recycle, regardless of whether the bottle or can is theirs or from another company . </s> <s> The company expects that the 100 percent collection goal by 2030 will be reached by working with industry partners, customers, consumers, and even local communities . </s> <s> Coca-Cola also highlights the importance of improving the packaging of its products, something that is already an ongoing process . </s> <s> With a goal of making all product packaging 100 percent recyclable worldwide, the company intends on building better bottles either by reducing the plastic content, using more recycled materials, or developing plant-based materials . </s> <s> By 2030, the company hopes to have made their packaging made of 50 percent recycled content . </s> <s> Sustainability Efforts By The Coca-Cola CompanyThe World Without Waste vision is great news coming from a company that has been selling its products in single-use plastic and can packaging for years . </s> <s> That said, this is not the first time that the company has made concrete steps in trying to be a more sustainable company . </s> <s> In fact, apart from the newly announced 2030 goals, it also has 2020 sustainability goals, including water stewardship goals which they already achieved in 2015, five years in advance . </s> <s> Also a part of its sustainability goals are climate protection, agriculture, packaging and recycling, and even matters such as giving back, human and workplace rights, and women's economic empowerment . </s> <s> According to the press release, the steps that the company is taking are a part of its strategy to grow the right way, that is, with conscience . </s> <s> In a statement by Coca-Cola Company president and CEO James Quincey, he states that the world's packaging problem is not just a problem for sea turtles and coastal communities but for all of us too, and that businesses like Coca-Cola must do more to create better packaging and dispose of them properly . </s> <s> ''When we all come together through concerted, collective action, we can make a meaningful, lasting difference . </s> <s> That must be the mindset we all take . </s> <s> The guiding principle by which we all operate, '' said Quincey . </s> <s>  . </s>  </p> </d>	publisher=AFP"
#
# for feature in sentence.strip().split('\t'):
#     print(feature)
#     (k, v) = feature.split('=')
#     print(k)
#     print(v)

# for i in range(2, 3):
#     out_path = './bytecup2018/traing-{}.txt'.format(i)
#     print(out_path)
#     out_file = open(out_path, 'w', encoding='utf-8')
#     in_path = './bytecup2018/bytecup.corpus.train.{}.txt'.format(i)
#     with open(in_path, 'r', encoding='utf-8') as file:
#         try:
#             for line in file.readlines():
#
#                 # 将str转换为dict
#                 data = json.loads(line.strip('\n'))
#                 article = data['content']
#                 # article = article.replace('\'', ' \\\'')
#                 article = article.replace('\"', ' \'\'')
#                 article = article.replace('\n', ' ')
#                 article = article.replace('\t', ' ')
#                 article = article.replace('=', ' ')
#
#                 result = ''
#                 sentence_list = article.split('.')
#                 for sent in sentence_list:
#                     result += '<s> %s . </s> ' % sent.strip()
#                 result = result.strip()
#                 article = 'article=<d> <p> %s  </p> </d>' % result
#                 abstract = data['title']
#                 abstract = abstract.replace('=', ' ')
#                 abstract = 'abstract=<d> <p> <s> %s </s> </p> </d>' % abstract
#                 publisher = 'publisher=AFP'
#                 sentence = abstract + '\t' + article + '\t' + publisher + '\n'
#                 print(sentence)
#                 out_file.write(sentence)
#                 # sentence = summarize(' ', data['content'])
#                 # sentence = parser.removePunctations(sentence)
#                 # sentence = parser.removeStopWords(list(sentence.split(' ')))
#
#                 # sentence = summarize(sentence, ratio=0.1, words=20)
#                 # out_file.write(sentence)
#                 # output.close()
#                 # print('summarzie:' + sentence)
#                 # print('len:', len(sentence))
#         finally:
#             file.close()
#     out_file.close()

# covert validation
out_path = './bytecup2018/valid.txt'
print(out_path)
out_file = open(out_path, 'w', encoding='utf-8')
in_path = './bytecup2018/bytecup.corpus.validation_set.txt'
with open(in_path, 'r', encoding='utf-8') as file:
    try:
        for line in file.readlines():

            # 将str转换为dict
            data = json.loads(line.strip('\n'))
            article = data['content']
            # article = article.replace('\'', ' \\\'')
            article = article.replace('\"', ' \'\'')
            article = article.replace('\n', ' ')
            article = article.replace('\t', ' ')
            article = article.replace('=', ' ')

            result = ''
            has_abstract = False
            abstract = ''
            sentence_list = article.split('.')
            for sent in sentence_list:
                sent = sent.strip()
                if has_abstract is False and len(sent) > 10:
                    abstract += sent
                    has_abstract = True
                result += '<s> %s . </s> ' % sent
            result = result.strip()
            article = 'article=<d> <p> %s  </p> </d>' % result
            # abstract = data['title']
            abstract = abstract.replace('=', ' ')
            abstract = 'abstract=<d> <p> <s> %s </s> </p> </d>' % abstract
            publisher = 'publisher=AFP'
            sentence = article + '\n'
            print(sentence)
            out_file.write(abstract + '\t' + publisher + '\t' + sentence)
            # sentence = summarize(' ', data['content'])
            # sentence = parser.removePunctations(sentence)
            # sentence = parser.removeStopWords(list(sentence.split(' ')))

            # sentence = summarize(sentence, ratio=0.1, words=20)
            # out_file.write(sentence)
            # output.close()
            # print('summarzie:' + sentence)
            # print('len:', len(sentence))
    finally:
        file.close()
out_file.close()

