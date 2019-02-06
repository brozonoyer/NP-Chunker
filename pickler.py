#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code to save and load train model
brozonoyer@brandeis.edu
"""


import pickle
from sklearn.externals import joblib

def load(model_path):

    try:
        print('Load model from {}'.format(model_path))
        '''Load model from path'''
        with open(model_path, 'rb') as fh:
            type_classifier = pickle.load(fh)

            return type_classifier
    except:
        return None


def save(self, model_path):
    try:
        with open(model_path, 'wb') as fh:
            pickle.dump(self, fh)
        print('Save model succeeded.')
    except:
        print('Save model failed.')