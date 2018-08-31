import os
import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from nltk import word_tokenize
from utils import confusion_matrix
from sklearn.model_selection import train_test_split

class NaiveBayes(object):

    def __init__(self):
        """
        Construct a new 'NavieBayes' object.
        :param log_class_priors: logarithimic of prior probability
        :param word_count: Number of word in true and fake news article
        :param vocabulary: Unique set of words in dictionary
        :return: returns nothing
        """
        self.log_class_priors={}
        self.words_count={'true':{},'fake':{}}
        self.vocabulary = set()

    def get_word_counts(self, words):
        """
        count the words from tokenize set of words
        :param words: list of tokenize words
        :return: dictionary with key as word and value as its associated count
        """
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0 #0.0 is defalult value in no is available
        return word_counts


    def set_vocabulary(self,word):
        """
        Add a word to vocabulary set
        :param word: word to be added
        :return: nothing is returned
        """
        if word not in self.vocabulary:
         self.vocabulary.add(word)

    def set_word_count(self,word,count,c):
        """
        Count word based on their labels
        :param word: word to be count
        :param count: value of word count
        :param c: classes for that word
        :return: nothing returned
        """
        if word not in self.words_count[c]:
            self.words_count[c][word]=0.0
        self.words_count[c][word]+=count




    def fit(self,X,Y):
        """
        Calculate the prior probability from word count
        :param X: training data set, news article text
        :param Y: label for training data set
        :return: no return only changes classes variable
        """
        n = len(X)
        #class prior probability
        self.log_class_priors['true'] = (sum(1 for label in Y if label == 0) / n)
        self.log_class_priors['fake'] =(sum(1 for label in Y if label == 1)/n)

        for x,y in zip(X,Y):
            c = 'true' if y == 0 else 'fake'
            counts = self.get_word_counts(word_tokenize(str(x)))
            for word,count in counts.items():
                self.set_vocabulary(word)
                self.set_word_count(word,count,c)


    def predict(self,X):
        """
        Predict whether a news article is true of false
        :param X: list of news article text data
        :return: predicated label
        """
        result = []
        for x in X:
            counts = self.get_word_counts(word_tokenize(str(x)))
            fake_prob = true_prob = 0
            for word,count in counts.items():
                if word not in self.vocabulary:
                    continue
                log_word_given_true = math.log((self.words_count['true'].get(word,0.0)+1)/((sum(self.words_count['true'].values())) +len(self.vocabulary)))
                log_word_given_false = math.log((self.words_count['fake'].get(word,0.0)+1)/((sum(self.words_count['fake'].values())) +len(self.vocabulary)))
                fake_prob +=log_word_given_false
                true_prob +=log_word_given_true
            fake_prob += self.log_class_priors['fake']
            true_prob += self.log_class_priors['true']
            result.append(0) if true_prob > fake_prob else result.append(1)
        print (result)
        return result


if __name__ == '__main__':
    df = pd.read_csv("./data/Train.csv")
    #taking 100 data for testing purpose only
    df = df.head(100)
    # Make training and test sets
    X_train, x_test, Y_train, y_test = train_test_split(df['text'],df['label'],test_size=0.33)
    nb = NaiveBayes()
    nb.fit(X_train,Y_train)
    pred = nb.predict(x_test)
    y_test = y_test.values.tolist()
    accuracy = sum(1 for i in range(len(pred)) if pred[i] == y_test[i]) / float(len(pred))
    print("{0:.4f}".format(accuracy))
    confusion_matrix(y_test,pred)

