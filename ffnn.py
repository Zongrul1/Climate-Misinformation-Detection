import json
from keras.layers import Dropout, Embedding, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dense, LSTM
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import numpy as np

stopwords = set(stopwords.words('english'))

def load_label(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as js:
        data_list = []
        label_list = []
        data = json.load(js)
        for k, v in data.items():
            data_list.append(str(v['text']))
            label_list.append(v['label'])
        return data_list, label_list


def load_nolabel(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as js:
        data_list = []
        data = json.load(js)
        for k, v in data.items():
            data_list.append(str(v['text']))
        return data_list

# tokenize
def pre_tokenize(train, dev, test):
    from nltk.tokenize import TweetTokenizer
    from nltk.corpus import stopwords

    tt = TweetTokenizer()
    stopwords = set(stopwords.words('english'))
    x_train = []
    x_dev = []
    x_test = []
    wnl = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for w in train:
        list_words = tt.tokenize(w)
        list_words = [w.lower() for w in list_words if isinstance(w, str) == True]
        list_words = [w for w in list_words if w.isalpha()]
        filtered_words = [w for w in list_words if not w in stopwords]
        filtered_words = [wnl.lemmatize(w,'n') for w in filtered_words]
        filtered_words = [wnl.lemmatize(w, 'v') for w in filtered_words]
        filtered_words = [stemmer.stem(w) for w in filtered_words]
        x_train.append(' '.join(filtered_words))
    for w in dev:
        list_words = tt.tokenize(w)
        list_words = [w.lower() for w in list_words if isinstance(w, str) == True]
        list_words = [w for w in list_words if w.isalpha()]
        filtered_words = [w for w in list_words if not w in stopwords]
        filtered_words = [wnl.lemmatize(w,'n') for w in filtered_words]
        filtered_words = [wnl.lemmatize(w, 'v') for w in filtered_words]
        filtered_words = [stemmer.stem(w) for w in filtered_words]
        x_dev.append(' '.join(filtered_words))
    for w in test:
        list_words = tt.tokenize(w)
        list_words = [w.lower() for w in list_words if isinstance(w, str) == True]
        list_words = [w for w in list_words if w.isalpha()]
        filtered_words = [w for w in list_words if not w in stopwords]
        filtered_words = [wnl.lemmatize(w,'n') for w in filtered_words]
        filtered_words = [wnl.lemmatize(w, 'v') for w in filtered_words]
        filtered_words = [stemmer.stem(w) for w in filtered_words]
        x_test.append(' '.join(filtered_words))
    return x_train, x_dev, x_test


def tokenize(train, dev, test):
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(oov_token="<UNK>", lower=True)
    tokenizer.fit_on_texts(train)
    x_train = tokenizer.texts_to_matrix(train, mode="tfidf")
    x_dev = tokenizer.texts_to_matrix(dev, mode="tfidf")
    x_test = tokenizer.texts_to_matrix(test, mode="tfidf")
    num = len(x_train[0])
    return x_train, x_dev, x_test, num

# output
def output_test(pred):
    i = 0
    dict = {}
    for p in pred:
        temp_dict = {"label": str(p)}
        dict["test-%d" % i] = temp_dict
        i += 1
    path = 'test-output.json'
    f = open(path, 'w')
    jsonData = json.dumps(dict)
    f.write(jsonData)


def output_dev(pred):
    i = 0
    dict = {}
    for p in pred:
        temp_dict = {"label": str(p)}
        dict["dev-%d" % i] = temp_dict
        i += 1
    path = 'dev-output.json'
    f = open(path, 'w')
    jsonData = json.dumps(dict)
    f.write(jsonData)

train_filepath = 'train.json'
train_data_list, train_label_list = load_label(train_filepath)
train_filepath = 'train_climate.json'
train_negative_list1, train_negative_label_list1 = load_label(train_filepath)
train_filepath = 'train_other.json'
train_negative_list2, train_negative_label_list2 = load_label(train_filepath)
dev_filepath = 'dev.json'
dev_data_list, dev_label_list = load_label(dev_filepath)
test_filepath = 'test-unlabelled.json'
test_data_list = load_nolabel(test_filepath)
train_data_list = train_data_list + train_negative_list1 + train_negative_list2
train_data_list, dev_data_list, test_data_list = pre_tokenize(train_data_list, dev_data_list,
                                                              test_data_list)
X_train, X_dev, X_test, num = tokenize(train_data_list, dev_data_list, test_data_list)
y_train = np.array(train_label_list + train_negative_label_list1 + train_negative_label_list2)
y_dev = np.array(dev_label_list)

# shuffle
np.random.seed(200)
np.random.shuffle(X_train)
np.random.seed(200)
np.random.shuffle(y_train)

#RNN
from keras.models import Sequential
from keras import layers, Input, Model

#model definition
model = Sequential(name="feedforward-bow-input")
model.add(Dense(10, input_dim=num, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#training
y_train = to_categorical(y_train,2)
y_dev = to_categorical(y_dev,2)
model.fit(X_train, y_train, epochs=10, verbose=True, validation_data=(X_dev, y_dev), batch_size=128)
pred = model.predict(X_test)
result = np.argmax(pred, axis=1)
print(result)
output_test(result)
pred = model.predict(X_dev)
result = np.argmax(pred, axis=1)
print(result)
output_dev(result)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("\nTesting Train Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_dev, y_dev, verbose=False)
print("\nTesting Dev Accuracy:  {:.4f}".format(accuracy))
