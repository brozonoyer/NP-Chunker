#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NP Chunker Module
brozonoyer@brandeis.edu
"""


import os, pickler, nltk, pycrfsuite
from nltk.corpus import ConllChunkCorpusReader
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from SynTagRus_NP_Chunk_Corpus_Reader import SynTagRusNPChunkCorpusReader as reader

class Chunker:

    # available models
    types = {'UNIGRAM', 'BIGRAM', 'TRIGRAM', 'SKLEARN_MAXENT',\
             'CRF', 'SVM'}

    def __init__(self, type):
        if(type.upper() not in self.types):
            print('This chunker type is not available!')
        self.type = type.upper()
        self.vec = DictVectorizer()
        self.hasher = FeatureHasher()


    def train(self, model_path=None):
        '''
        Trains the chunker's model
        '''

        # path to store trained models
        if(model_path == None):
            model_path = os.path.sep.join(['..', 'Models'])

        if(self.type == 'UNIGRAM' or self.type == 'BIGRAM' or self.type == 'TRIGRAM'):

            #train_data = self.train_data()
            train_data = self.data("train")

            if(self.type=='UNIGRAM'):
                tagger = nltk.UnigramTagger(train_data, backoff=nltk.DefaultTagger('O'))
                pickler.save(tagger, os.path.sep.join([model_path, 'UNIGRAM.pkl']))
            elif (self.type == 'BIGRAM'):
                tagger = nltk.BigramTagger(train_data, backoff=pickler.load(os.path.sep.join([model_path, 'UNIGRAM.pkl'])))
                pickler.save(tagger, os.path.sep.join([model_path, 'BIGRAM.pkl']))
            else:  # trigram
                tagger = nltk.TrigramTagger(train_data, backoff=pickler.load(os.path.sep.join([model_path, 'BIGRAM.pkl'])))
                pickler.save(tagger, os.path.sep.join([model_path, 'TRIGRAM.pkl']))

        else:

            #X_train, y_train = self.train_data()
            X_train, y_train = self.data("train")

            if(self.type == 'CRF'):
                import pycrfsuite

                trainer = pycrfsuite.Trainer(verbose=False)
                trainer.append(X_train, y_train)
                trainer.train(os.path.sep.join([model_path, 'CRF.pkl']))

            else: # sklearn_maxent or svm

                if(self.type == 'SKLEARN_MAXENT'):
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression()
                else: #svm
                    from sklearn.svm import LinearSVC
                    clf = LinearSVC()

                clf.fit(X_train, y_train)
                pickler.save(clf, os.path.sep.join([model_path, self.type+'.pkl']))


    def test(self, model_path=None):
        '''
        Decodes model.
        Evaluates the model and prints out classification report.
        Unfortunately, this is currently within the same method.
        '''

        from sklearn.metrics import classification_report

        # path to store trained models
        if (model_path == None):
            model_path = os.path.sep.join(['..', 'Models'])

        # necessary information for classification report
        labels_nums = {'B': 0, 'I': 1, 'O': 2, 'BH': 3, 'IH': 4}
        target_names = ['B', 'I', 'O', 'BH', 'IH']

        # used for classification report
        pred = []
        truth = []

        if(self.type == 'UNIGRAM' or self.type == 'BIGRAM' or self.type == 'TRIGRAM'):

            #tagged_sents, gold = self.test_data()
            tagged_sents, gold = self.data("test")

            if(self.type == 'UNIGRAM'):
                tagger = pickler.load(os.path.sep.join([model_path, 'UNIGRAM.pkl']))
            elif (self.type == 'BIGRAM'):
                tagger = pickler.load(os.path.sep.join([model_path, 'BIGRAM.pkl']))
            else:  # Trigram
                tagger = pickler.load(os.path.sep.join([model_path, 'TRIGRAM.pkl']))

            # go through each sentence in test_sentences and extend prediction list with the tagged sentence
            for s in tagged_sents:
                pos_tags = [pos for (w, pos) in s]                      # pos for sentence
                pred.extend([c for (pos, c) in tagger.tag(pos_tags)])   # iob prediction for pos

            # go through each gold sentence and get true iob chunk label
            for s in gold:
                truth.extend([i for (i, _, _, _, _) in s])

        else:

            #X_test, truth = self.test_data()
            X_test, truth = self.data("test")

            if(self.type == 'CRF'):
                import pycrfsuite

                clf = pycrfsuite.Tagger()
                clf.open(model_path + os.path.sep + 'CRF.pkl')
                print('Load model from', model_path + os.path.sep + 'CRF.pkl')
                pred = clf.tag(X_test)

            else:  # sklearn_maxent or svm
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import LinearSVC

                clf = pickler.load(os.path.sep.join([model_path, self.type+'.pkl']))
                pred = clf.predict(X_test)

        # convert iob labels to corresponding digit
        pred = [labels_nums[str(p)] for p in pred]
        truth = [labels_nums[t] for t in truth]

        print(classification_report(truth, pred, target_names=target_names))



    def data(self, mode, data_path=None):

        if(data_path==None):
            data_path = '..' + os.path.sep + '2_SynTagRus_NP_Chunks'
        fileids = [mode + ".txt"]

        # create reader for training data
        r = reader(data_path, fileids)
        sentences = r.iob_sents()

        # if train mode
        if(mode == "train"):

            if (self.type == 'UNIGRAM' or self.type == 'BIGRAM' or self.type == 'TRIGRAM'):

                train_data = [[(p, i) for i, w, l, p, f in sent]
                              for sent in sentences]
                return train_data

            else:
                from feature_extraction import Feature_Extractor

                fe_train = Feature_Extractor(sentences)
                train_features = fe_train.features()
                y_train = fe_train.labels()

                if (self.type == 'CRF'):
                    X_train = pycrfsuite.ItemSequence(train_features)
                else:  # sklearn
                    # X_train = self.vec.fit_transform(train_features)
                    X_train = self.hasher.fit_transform(train_features)

                return X_train, y_train

        # if test mode
        else:

            if (self.type == 'UNIGRAM' or self.type == 'BIGRAM' or self.type == 'TRIGRAM'):
                tagged_sents = r.tagged_sents()
                gold = sentences
                return tagged_sents, gold

            else:
                from feature_extraction import Feature_Extractor

                fe_test = Feature_Extractor(sentences)
                test_features = fe_test.features()
                y_test = fe_test.labels()

                if (self.type == 'CRF'):
                    X_test = pycrfsuite.ItemSequence(test_features)
                else:  # sklearn
                    # X_test = self.vec.fit_transform(test_features)
                    X_test = self.hasher.fit_transform(test_features)

                return X_test, y_test