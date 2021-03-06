import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import json
import pickle
import os.path
import bot

with open("samples.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) # 'how are you' -> ['how', 'are', 'you']
            words.extend(wrds) # ['how', 'are', 'you'] -> ['how', 'are', 'you'] 1 | ['hi', 'there'] -> ['how', 'are', 'you', 'hi', 'there'] 2
            docs_x.append(wrds) # ['how', 'are', 'you'] -> [['how', 'are', 'you']] 1 | ['hi', 'there'] -> [['how', 'are', 'you'], ['hi', 'there']] 2
            docs_y.append(intent["tag"]) # 'greeting' -> ['greeting', 'greeting']

        if intent["tag"] not in labels:
            labels.append(intent["tag"]) # 'thanks' -> ['greeting', 'thanks']

    words = [stemmer.stem(w.lower()) for w in words if w != "?"] # ['how', 'are', 'you', 'grater'] -> ['how', 'ar', 'you', 'grate']
    
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


if os.path.exists("model.tflearn.index"):
    model.load("model.tflearn", weights_only=True)
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

bot.chat(model, (words, labels))
