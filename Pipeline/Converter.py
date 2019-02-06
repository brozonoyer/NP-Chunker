#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module that converts dependency trees to conll-style files
brozonoyer@brandeis.edu
"""



import nltk, random, os
from nltk.corpus import DependencyCorpusReader as dcr
from copy import deepcopy

np_components = {'A', 'NUM', 'S', 'COM'}

def path(g): #g: graph
    """
    Adapted from syntagrus_projectivity.py from https://github.com/luutuntin/SynTagRus_DS2PS

    -> specific sequence of all the nodes in g
    such that any dependent node comes before its head
    """
    marked = set()
    nodes = set(g.nodes)
    output = list()
    def recursive(g):
        for i in nodes.copy():
            d = [addr for (addr,_) in dependents(g, g.nodes[i])]
            if (not d) or all(dd in marked for dd in d):
                output.append((i,g.nodes[i]['word']))
                marked.add(i)
                nodes.remove(i)
                if nodes==set([0]):
                    break
                recursive(g)
                break
    recursive(g)
    return output

def change_head(g, n):  # g : graph, n : node
    if (n['tag'] == 'S'):  # noun
        if (((get_case(n) not in {'РОД', '_'}) and  # if not genitive
                 (g.nodes[n['head']]['tag'] != 'S' or n['feats'] !=
                     g.nodes[n['head']]['feats']))  # if not agreed with head N
            or (get_case(n) == 'РОД' and g.nodes[n['head']][
                'tag'] != 'S')):  # independent genitive

            old_head_addr = n['head']
            old_rel = n['rel']
            n['head'] = 0
            n['rel'] = 'ROOT'
            g.nodes[old_head_addr]['deps'][old_rel].remove(n['address'])
            g.add_arc(0, n['address'])


def get_case(n):  # n : node
    feats_dict = get_features(n)
    if ('case' in feats_dict.keys()):
        return feats_dict['case']
    return '_'


def get_features(n):
    feats_dict = {}
    if n['feats'] and (n['feats'] not in {'_', 'None'}):
        feats = n['feats'].split('|')
        for f in feats:
            feat_val = f.split('=')
            feat = feat_val[0]
            val = feat_val[1]
            feats_dict[feat] = val
    return feats_dict


def dependents(g, n):  # n: node
    """ -> sorted list of dep addresses """
    output = []

    if (n['deps'] != {}):
        for rel in n['deps']:
            if (n['deps'][rel] != []):
                for addr in n['deps'][rel]:
                    output.append((addr, rel))
    return sorted(output)


def remove_dependents(g, n):
    '''remove dependents from node which can't belong to base np'''
    deps = dependents(g, n)
    for (addr, rel) in deps:
        d = g.nodes[addr]  # get dependent node
        if (d[
                'tag'] not in np_components):  # if node can't belong to base NP, remove from node's dependency list
            n['deps'][rel].remove(addr)


def label_base_np_heads(g):
    root = g.nodes[0]

    if (dependents(g, root) != []):
        np_index = 0
        for d in dep_nodes(g, root):
            if (d['tag'] == 'S'):
                d['np'] = np_index
                d['iob'] = 'H'
                if (dependents(g, d) != []):
                    for dd in dep_nodes(g, d):
                        assign_np_index(g, dd, np_index)
                np_index = np_index + 1
    return np_index


def assign_np_index(g, n, np_index):
    n['np'] = np_index
    if (dependents(g, n) != []):
        for d in dep_nodes(g, n):
            assign_np_index(g, d, np_index)


def dep_nodes(g, n):
    return [g.nodes[addr] for (addr, _) in dependents(g, n)]


def assign_iob_labels(g, num_nps):  # g: graph, num_nps: number of nps in sentence
    nodes = g.nodes
    for i in range(num_nps):
        np_indices = [j for j in range(len(nodes)) if \
                      (('np' in nodes[sorted(nodes)[j]]) and (
                      nodes[sorted(nodes)[j]]['np'] == i))]
        if (np_indices[-1] - np_indices[0] + 1 > len(np_indices)):
            return False  # eliminate sentence with nested base_nps
        np = [nodes[sorted(nodes)[j]] for j in np_indices]
        if ('iob' in np[0]):
            np[0]['iob'] = 'BH'
        else:
            np[0]['iob'] = 'B'
        if (len(np) > 1):
            for node in np[1:]:
                if ('iob' in node):
                    node['iob'] = 'IH'
                else:
                    node['iob'] = 'I'
    for addr in nodes:
        if ('iob' not in nodes[addr]):
            nodes[addr]['iob'] = 'O'
    return True


def convert2conll(g):
    output = []
    for addr in sorted(g.nodes):
        if (addr != 0 and not g.nodes[addr]['word'].startswith('*NP2P*')):
            n = g.nodes[addr]
            line = " ".join([
                n['iob'],
                n['word'],
                n['lemma'],
                n['tag'],
                n['feats'],
            ])
            output.append(line)
    return output


def split(sents, test_size=0.20):
    random.seed()
    random.shuffle(sents)
    test_size = int(len(sents) * test_size)
    test = sents[:test_size]
    train = sents[test_size:]
    return train, test

if __name__ == '__main__':

    # dependency tree reader
    reader = dcr(os.path.sep.join(['..', '1_SynTagRus_Projective']),
                 ['projectivized-syntagrus_conll2007.txt'],
                 encoding='windows-1251')
    parsed_sents = reader.parsed_sents()
    copy = [deepcopy(s) for s in parsed_sents] # work with copy

    conll_sents = []

    # convert to conll-style sentences
    for g in copy:
        for (addr, _) in path(g):
            change_head(g, g.nodes[addr])
        for n in dep_nodes(g, g.nodes[0]):
            if (n['tag'] == 'S'):
                remove_dependents(g, n)
        num_nps = label_base_np_heads(g)
        if (assign_iob_labels(g, num_nps)):
            conll_sents.append(convert2conll(g))

    # split into testing and training sentences
    train, test = split(conll_sents)

    # save train data to 2_SynTagRus_NP_Chunks folder
    with open(os.path.sep.join(['..', '2_SynTagRus_NP_Chunks', 'train.txt']), 'w',
              encoding='windows-1251') as f:
        f.write('\n\n'.join(['\n'.join(sent) for sent in train]))

    # save test data to 2_SynTagRus_NP_Chunks folder
    with open(os.path.sep.join(['..', '2_SynTagRus_NP_Chunks', 'test.txt']), 'w',
              encoding='windows-1251') as f:
        f.write('\n\n'.join(['\n'.join(sent) for sent in test]))