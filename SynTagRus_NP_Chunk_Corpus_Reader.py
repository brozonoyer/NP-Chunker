#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reader for converted data
Adapted from nltk's ConllCorpusReader
brozonoyer@brandeis.edu
"""



from six import string_types

from nltk import compat
from nltk.util import LazyMap, LazyConcatenation
from nltk.tag import map_tag

from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import *

class SynTagRusNPChunkCorpusReader(CorpusReader):
    """
    Adapted from http://www.nltk.org/_modules/nltk/corpus/reader/conll.html#ConllCorpusReader

    A corpus reader for CoNLL-style files.  These files consist of a
    series of sentences, separated by blank lines.  Each sentence is
    encoded using a table (or "grid") of values, where each line
    corresponds to a single word, and each column corresponds to an
    annotation type.  The set of columns used by CoNLL-style files can
    vary from corpus to corpus; the ``ConllCorpusReader`` constructor
    therefore takes an argument, ``columntypes``, which is used to
    specify the columns that are used by a given corpus.

    """

    #/////////////////////////////////////////////////////////////////
    # Column Types
    #/////////////////////////////////////////////////////////////////

    IOB = 'iob'       #: column type for iob tags
    WORD = 'word'     #: column type for word forms
    LEMMA = 'lemma'   #: column type for lemmas
    POS = 'pos'     #: column type for part-of-speech tags
    FEATS = 'feats'   #: column type for other morphological features

    #: A list of all column types supported by this corpus reader.
    COLUMN_TYPES = (IOB, WORD, LEMMA, POS, FEATS)

    #/////////////////////////////////////////////////////////////////
    # Constructor
    #/////////////////////////////////////////////////////////////////

    def __init__(self, root, fileids, encoding='windows-1251', tagset=None):
        columntypes = ('iob','word', 'lemma', 'pos', 'feats')
        self._colmap = dict((c,i) for (i,c) in enumerate(columntypes))
        CorpusReader.__init__(self, root, fileids, encoding)
        self._tagset = tagset

    #/////////////////////////////////////////////////////////////////
    # Data Access Methods
    #/////////////////////////////////////////////////////////////////

    def raw(self, fileids=None):
        if fileids is None: fileids = self._fileids
        elif isinstance(fileids, string_types): fileids = [fileids]
        return concat([self.open(f).read() for f in fileids])


    def words(self, fileids=None):
        self._require(self.WORD)
        return LazyConcatenation(LazyMap(self._get_words, self._grids(fileids)))


    def sents(self, fileids=None):
        self._require(self.WORD)
        return LazyMap(self._get_words, self._grids(fileids))


    def tagged_words(self, fileids=None, tagset=None):
        self._require(self.WORD, self.POS)
        def get_tagged_words(grid):
            return self._get_tagged_words(grid, tagset)
        return LazyConcatenation(LazyMap(get_tagged_words,
                                         self._grids(fileids)))


    def tagged_sents(self, fileids=None, tagset=None):
        self._require(self.WORD, self.POS)
        def get_tagged_words(grid):
            return self._get_tagged_words(grid, tagset)
        return LazyMap(get_tagged_words, self._grids(fileids))


    def iob_words(self, fileids=None, tagset=None):
        """
        :return: a list of iob/word/lemma/pos/feats tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.IOB, self.WORD,self.LEMMA, self.POS, self.FEATS)
        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)
        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))


    def iob_sents(self, fileids=None, tagset=None):
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.IOB, self.WORD,self.LEMMA, self.POS, self.FEATS)
        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)
        return LazyMap(get_iob_words, self._grids(fileids))


    #/////////////////////////////////////////////////////////////////
    # Grid Reading
    #/////////////////////////////////////////////////////////////////

    def _grids(self, fileids=None):
        # n.b.: we could cache the object returned here (keyed on
        # fileids), which would let us reuse the same corpus view for
        # different things (eg srl and parse trees).
        return concat([StreamBackedCorpusView(fileid, self._read_grid_block,
                                              encoding=enc)
                       for (fileid, enc) in self.abspaths(fileids, True)])

    def _read_grid_block(self, stream):
        grids = []
        for block in read_blankline_block(stream):
            block = block.strip()
            if not block: continue

            grid = [line.split() for line in block.split('\n')]

            # If there's a docstart row, then discard. ([xx] eventually it
            # would be good to actually use it)
            if grid[0][self._colmap.get('word', 0)] == '-DOCSTART-':
                del grid[0]

            # Check that the grid is consistent.
            for row in grid:
                if len(row) != len(grid[0]):
                    raise ValueError('Inconsistent number of columns:\n%s'
                                     % block)
            grids.append(grid)
        return grids

    #/////////////////////////////////////////////////////////////////
    # Transforms
    #/////////////////////////////////////////////////////////////////
    # given a grid, transform it into some representation (e.g.,
    # a list of words or a parse tree).

    def _get_words(self, grid):
        return self._get_column(grid, self._colmap['word'])

    def _get_tagged_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['word']), pos_tags))

    def _get_iob_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['iob']),
                        self._get_column(grid, self._colmap['word']),
                        self._get_column(grid, self._colmap['lemma']),
                        pos_tags,
                        self._get_column(grid, self._colmap['feats'])))

    #/////////////////////////////////////////////////////////////////
    # Helper Methods
    #/////////////////////////////////////////////////////////////////

    def _require(self, *columntypes):
        for columntype in columntypes:
            if columntype not in self._colmap:
                raise ValueError('This corpus does not contain a %s '
                                 'column.' % columntype)

    @staticmethod
    def _get_column(grid, column_index):
        return [grid[i][column_index] for i in range(len(grid))]