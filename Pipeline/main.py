#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main method: run models here
brozonoyer@brandeis.edu
"""



from Pipeline.Chunker import Chunker

if __name__ == '__main__':

    #pass

    u = Chunker('unigram')
    u.train()
    u.test()

    #b = Chunker('bigram')
    #b.train()
    #b.test()

    #t = Chunker('trigram')
    #t.train()
    #t.test()

    #maxent = Chunker('sklearn_maxent')
    #maxent.train()
    #maxent.test()

    #svm = Chunker('svm')
    #svm.train()
    #svm.test()

    crf = Chunker('crf')
    crf.train()
    crf.test()