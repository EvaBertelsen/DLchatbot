
## imports ##

import nltk # natural language toolkit that translates human language data to python's programming language

from nltk.stem.lancaster import LancasterStemmer ## stemmers are from nltk that removes gramatical affixes ("!", "?", "," , ".", etc) so it's easier to translate
stemmer = LancasterStemmer () # lancaster is just a specific nlt library with stemmers

import numpy

import tensorflow # deeplearning api

import tflearn # deeplearning library that works with Tensorflow

import random

import json # to be able to read intents.json


# associates the body.py with intents.json 

with open("intents.json") as file: 
    data =json.load(file)


# variables

words = []
labels = []
docs_x = []
docs_y = []



for intent in data ["intents"]: 
    for pattern in intent ["patterns"]:
        wrds = nltk.word_tokenize(pattern) 
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])
# asks to check the data from intents.json 
# and checks each pattern that is introduced. 
# this is where stemming starts. it removes the grammatical affixes whenever there's an input, so it can be readable to the machine 
# the next step in this for loop is to tokenize it (which )

    if intent ["tag"] not in labels: ## from the json file greetings, goodbye, problem, etc
        labels.append(intent["tag"]) 
        # This if statement is if there's not a generalized input it will re-try and re-us any of the input under "tag" as output.   
 
words = [stemmer.stem(w.lower()) for w in words if w != "?"] # so whenever there's an unexpected error or something not found within json as response "?"

words = sorted(list(set(words)))

#training model set output for the machine

# we will assign 0 and 1 for words to train the neural network to distinguish what exists and not exist within the json file

training = []

output = []

out_empty = [0 for _ in range (len(labels))] 

for x, doc in enumerate (docs_x):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

    for w in words: 
        if w in wrds:
            bag.append(1) # 1 represents this word exists
        else: 
            bag.append(0) # if the word doesnt exist translate 0
    
    output_row = list(out_empty[:])
    output_row[labels.index(docs_y[x])] = 1 

    training.append(bag)
    output.append(output_row)

training = numpy.array(training) # now we are using the numpy import from above 
output = numpy.array(output)


#Model building

net = tflearn.input_data(shape=[None, len(training[0])]) # this is going to find the input shape that is expected wth the length of training 0
net = tflearn.fully_connected (net, 42)
net = tflearn.fully_connected (net, 42)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


# softmax or softargmax is a normalized exponentional function
# within the input  each input is connected and will lead to a connected output, similar to a decision tree diagram 
# this connected output can be any of the tags (hello, good bye, problem, etc)