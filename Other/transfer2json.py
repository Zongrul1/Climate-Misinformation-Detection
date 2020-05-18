import json
import os
from nltk.tokenize import TweetTokenizer

tt = TweetTokenizer()
data_list = []
label_list = []
# for filename in os.listdir("news\\"):
#     for file in os.listdir("news\\" + filename):
#         with open("news\\" + filename + "\\" + file, 'r', encoding='ISO-8859-1') as f:
#             text = ''
#             for ff in f.readlines():
#                 text += ff.strip()
#             data_list.append(text)
#             label_list.append(0)
j = 0
with open("neg_train.json", 'r', encoding='ISO-8859-1') as js:
    data = json.load(js)
    for k, v in data.items():
        list_words = tt.tokenize(str(v['text']))
        if j > 1000:
            break
        if (len(list_words) > 25):
            data_list.append(str(v['text']))
            label_list.append(v['label'])
        j += 1
i = 0
dict = {}
for i in range(len(data_list)):
    temp_dict = {"text": str(data_list[i]),"label":str(label_list[i])}
    dict["train-%d" % i] = temp_dict
    i += 1
path = 'train_climate.json'
f = open(path, 'w')
jsonData = json.dumps(dict)
f.write(jsonData)