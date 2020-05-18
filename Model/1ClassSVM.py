import json
import nltk
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))


def load(filepath):
    with open(filepath, 'r', encoding='UTF-8') as js:
        data_list = []
        label_list = []
        data = json.load(js)
        for k, v in data.items():
            data_list.append(v['text'])
            label_list.append(v['label'])
        return data_list, label_list


def loadun(filepath):
    with open(filepath, 'r', encoding='UTF-8') as js:
        data_list = []
        data = json.load(js)
        for k, v in data.items():
            data_list.append(v['text'])
        return data_list


# tokenize
def tokenize(data_list):
    from nltk.tokenize import TweetTokenizer
    from nltk.corpus import stopwords

    tt = TweetTokenizer()
    stopwords = set(stopwords.words('english'))
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    climate_list = []
    for d in data_list:
        data_dict = {}
        list_words = tt.tokenize(d)
        list_words = [w.lower() for w in list_words if isinstance(w, str) == True]
        list_words = [w for w in list_words if w.isalpha()]
        filtered_words = [w for w in list_words if not w in stopwords]
        wnl = nltk.WordNetLemmatizer()
        filtered_words = [wnl.lemmatize(w, 'n') for w in filtered_words]
        filtered_words = [wnl.lemmatize(w, 'v') for w in filtered_words]
        stemmer = nltk.PorterStemmer()
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
train_data_list, train_label_list = load(train_filepath)
X_train = tokenize(train_data_list)
dev_filepath = 'dev.json'
dev_data_list, dev_label_list = load(dev_filepath)
X_dev = tokenize(dev_data_list)
test_filepath = 'test-unlabelled.json'
test_data_list = loadun(test_filepath)
X_test = tokenize(test_data_list)
# vectorize
from sklearn.feature_extraction import DictVectorizer

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_dev = vectorizer.transform(X_dev)
X_test = vectorizer.transform(X_test)

# tf-idf
from sklearn import svm
# model
clf = svm.OneClassSVM(nu=0.25, kernel="rbf")
clf.fit(X_train)
pred = clf.predict(X_dev)
print(pred)
i = 0
pred = pred.tolist()
for i in range(len(pred)):
    if pred[i] == -1:
        pred[i] = 0
output(pred)
