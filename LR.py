import json

# load
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, LancasterStemmer, PorterStemmer

stopwords = set(stopwords.words('english'))

def load_label(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as js:
        data_list = []
        label_list = []
        data = json.load(js)
        for k, v in data.items():
            data_list.append(v['text'])
            label_list.append(v['label'])
        return data_list, label_list


def load_nolabel(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as js:
        data_list = []
        data = json.load(js)
        for k, v in data.items():
            data_list.append(v['text'])
        return data_list

from nltk.corpus import wordnet

#pos
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default noun

# tokenize for dev
def tokenize(data_list):
    from nltk.tokenize import TweetTokenizer
    from nltk.corpus import stopwords

    tt = TweetTokenizer()
    stopwords = set(stopwords.words('english'))
    climate_list = []
    for d in data_list:
        d = str(d)
        data_dict = {}
        list_words = tt.tokenize(d)
        list_words = [w.lower() for w in list_words if isinstance(w, str) == True]
        list_words = [w for w in list_words if w.isalpha()]
        filtered_words = [w for w in list_words if not w in stopwords]
        wnl = WordNetLemmatizer()
        filtered_words = [wnl.lemmatize(w, 'n') for w in filtered_words]
        filtered_words = [wnl.lemmatize(w, 'v') for w in filtered_words]
        stemmer = PorterStemmer()
        filtered_words = [stemmer.stem(w) for w in filtered_words]
        for i in filtered_words:
            key = data_dict.get(i)
            if key == None:
                data_dict[i] = 1
            else:
                data_dict[i] += 1
        climate_list.append(data_dict)
    return climate_list

# output
def output(pred):
    i = 0
    dict = {}
    for p in pred:
        temp_dict = {"label": str(p)}
        dict["dev-%d" % i] = temp_dict
        i += 1
    path = 'dev-output.json'
    # print(dict)
    f = open(path, 'w')
    jsonData = json.dumps(dict)
    f.write(jsonData)


# main-preprocess
train_filepath = 'train.json'
train_data_list, train_label_list = load_label(train_filepath)
train_data_list = tokenize(train_data_list)
train_filepath = 'train_climate.json'
train_negative_data_list1, train_negative_label_list1 = load_label(train_filepath)
train_negative_data_list1 = tokenize(train_negative_data_list1)
train_filepath = 'train_other.json'
train_negative_data_list2, train_negative_label_list2 = load_label(train_filepath)
train_negative_data_list2 = tokenize(train_negative_data_list2)
dev_filepath = 'dev.json'
dev_data_list, dev_label_list = load_label(dev_filepath)
dev_data_list = tokenize(dev_data_list)
test_filepath = 'test-unlabelled.json'
test_data_list = load_nolabel(test_filepath)
test_data_list = tokenize(test_data_list)

# train
X_train = []
y_train = []
X_dev = []
y_dev = []
X_test = []
for d in train_data_list:
    X_train.append(d)
for l in train_label_list:
    y_train.append(l)
for d in train_negative_data_list1:
    X_train.append(d)
    y_train.append(0)
for d in train_negative_data_list2:
    X_train.append(d)
    y_train.append(0)
for d in dev_data_list:
    X_dev.append(d)
for l in dev_label_list:
    y_dev.append(l)
for d in test_data_list:
    X_test.append(d)

# vectorize
from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)
X_test = vectorizer.transform(X_test)

# tf-idf
from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=True)
X_train = transformer.fit_transform(X_train)

# model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1.0, penalty='l2',solver='liblinear')
pred = model.fit(X_train, y_train).predict(X_dev)
output(pred)
print(pred)

