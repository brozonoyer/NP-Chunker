#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class to extract features from data
brozonoyer@brandeis.edu
"""


def get_features(feats_string):
    feats_dict = {}
    if (feats_string not in {'_', 'None', '<START>', '<END>'}):
        feats = feats_string.split('|')
        for f in feats:
            feat_val = f.split('=')
            feat = feat_val[0]
            val = feat_val[1]
            feats_dict[feat] = val
    return feats_dict


def get_feat(feats_dict, feat_name):
    if(feat_name in feats_dict):
        return feats_dict[feat_name]
    return 'N/A'


class Feature_Extractor:

    def __init__(self, sentences):
        #self.data = [[((w, p), i) for i, w, l, p, f in sent] for sent in sentences]
        self.data = sentences


    def words(self):
        '''
        :return: words in data
        '''
        words = [[w for (_, w, _, _, _) in sent] for sent in self.data]
        return words


    def features(self):
        '''
        :return: feature dict with more features (X_sequence)
        You should edit the feature_dict that is returned to manually adjust the training features
        '''
        feature_dict = []
        for sentence in self.data:
            for i in range(len(sentence)):
                _, word, lemma, pos, feats = sentence[i]
                if i == 0:
                    prevword=prevlemma=prevpos=prevfeats = "<START>"
                else:
                    _, prevword, prevlemma, prevpos, prevfeats = sentence[i - 1]
                if i == 0 or i == 1:
                    prevprevword = prevprevlemma = prevprevpos = prevprevfeats = "<START>"
                else:
                    _, prevprevword, prevprevlemma, prevprevpos, prevprevfeats = sentence[i - 2]
                if i == len(sentence) - 1:
                    nextword=nextlemma=nextpos=nextfeats = "<END>"
                else:
                    _, nextword, nextlemma, nextpos, nextfeats = sentence[i + 1]

                morph_feats = get_features(feats)
                prev__morph_feats = get_features(prevfeats)
                next_morph_feats = get_features(nextfeats)

                # edit the features here
                feature_dict.append({"pos": pos,
                                     #"word": word,
                                     #"prevword": prevword,
                                     #"prevpos": prevpos,
                                     #prevprevpos": prevprevpos
                                     #"nextword": nextword,
                                     #"nextpos": nextpos,
                                     #"prevpos+pos": "%s+%s" % (prevpos, pos),
                                     #"pos+nextpos": "%s+%s" % (pos, nextpos),
                                     #"lemma": lemma,
                                     #"gender": get_feat(morph_feats, 'gender'),
                                     #"case": get_feat(morph_feats, 'case'),
                                     #"prevcase": get_feat(prev__morph_feats, 'case'),
                                     #"nextcase": get_feat(next_morph_feats, 'case'),
                                     #"number": get_feat(morph_feats, 'number'),
                                     #"animacy": get_feat(morph_feats, 'animacy')
                                     })
        return feature_dict


    def labels(self):
        '''
        :return: IOB label (y_sequence)
        '''
        labels = [(i) for sent in self.data for (i, _, _, _, _) in sent]
        return labels


