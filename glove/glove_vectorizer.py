import collections

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin, TfidfVectorizer
from joblib import Parallel, delayed, cpu_count

from .corpus import Corpus
from .glove import Glove, check_random_state
from .glove_cython import transform_paragraph

PRETRAINED_DICT = {
    50: 'glove.6B.50d.txt',
    100: 'glove.6B.100d.txt',
    200: 'glove.6B.200d.txt',
    300: 'glove.6B.300d.txt',
}


def _transform_paragraph(paragraph, glove, epochs=50, random_state=None,
                         ignore_missing=False):
    if glove.word_vectors is None:
        raise Exception('Model must be fit to transform paragraphs')

    if glove.dictionary is None:
        raise Exception('Dictionary must be provided to '
                        'transform paragraphs')

    cooccurrence = collections.defaultdict(lambda: 0.0)

    for token in paragraph:
        try:
            cooccurrence[glove.dictionary[token]] += glove.max_count / 10.0
        except KeyError:
            if not ignore_missing:
                raise

    random_state = check_random_state(random_state)

    word_ids = np.array(cooccurrence.keys(), dtype=np.int32)
    values = np.array(cooccurrence.values(), dtype=np.float64)
    shuffle_indices = np.arange(len(word_ids), dtype=np.int32)

    # Initialize the vector to mean of constituent word vectors
    paragraph_vector = np.mean(glove.word_vectors[word_ids], axis=0)
    sum_gradients = np.ones_like(paragraph_vector)

    # Shuffle the coocurrence matrix
    random_state.shuffle(shuffle_indices)
    transform_paragraph(glove.word_vectors,
                        glove.word_biases,
                        paragraph_vector,
                        sum_gradients,
                        word_ids,
                        values,
                        shuffle_indices,
                        glove.learning_rate,
                        glove.max_count,
                        glove.alpha,
                        epochs)

    return paragraph_vector

class GloveVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', vocabulary=None,
                 window=10, n_components=100, n_jobs=-1, learning_rate=0.05,
                 n_glove_epochs=10, n_paragraph_epochs=50,
                 pre_trained=False, random_state=None,  verbose=False):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words

        self.window = window
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_glove_epochs = n_glove_epochs
        self.n_paragraph_epochs = n_paragraph_epochs
        self._n_jobs = n_jobs
        self.pre_trained = pre_trained
        self.random_state = random_state
        self.verbose = verbose

        self.glove = None

    @property
    def vocabulary_(self):
        if self.glove is None:
            raise ValueError('Model not fitted.')
        return self.glove.dictionary

    @property
    def n_jobs(self):
        if self._n_jobs < 0:
            return cpu_count()
        return min(self._n_jobs, cpu_count())

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):
        random_state = check_random_state(self.random_state)

        raw_documents = self._apply_analyzer(raw_documents)

        if self.pre_trained:
            # get the pre-trained glove vectors
            if self.n_components in PRETRAINED_DICT:
                pretrained_vectors = PRETRAINED_DICT[self.n_components]
            else:
                raise ValueError('No pretrained glove representation with '
                                 'n_components = {}'.format(self.n_components))
            self.glove = Glove.load_stanford(pretrained_vectors)
            self.glove.random_state = random_state
            self.fixed_vocabulary_ = True
        else:
            corpus = Corpus().fit(raw_documents, window=self.window)
            self.glove = Glove(no_components=self.n_components,
                               learning_rate=self.learning_rate,
                               random_state=random_state)
            self.glove.fit(corpus.matrix, epochs=self.n_glove_epochs,
                           no_threads=self.n_jobs, verbose=self.verbose)
            self.glove.add_dictionary(corpus.dictionary)

        return self._apply_glove(raw_documents, apply_analyzer=False)

    def transform(self, raw_documents):
        if self.glove is None:
            raise RuntimeError('Model glove must be fit first.')

        return self._apply_glove(raw_documents)

    def _apply_analyzer(self, raw_documents):
        analyze = self.build_analyzer()
        return [analyze(doc) for doc in raw_documents]

    def _apply_glove(self, raw_documents, apply_analyzer=True):
        if apply_analyzer:
            raw_documents = self._apply_analyzer(raw_documents)

        result = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(_transform_paragraph)(
                doc, self.glove, epochs=self.n_paragraph_epochs,
                random_state=self.random_state, ignore_missing=True)
            for doc in raw_documents)

        doc_vecs = np.array(result)

        return doc_vecs

    def most_similar(self, word, number=5):
        """
        Run similarity query, retrieving number of most similar words.
        """
        return self.glove.most_similar(word, number=number)

    def most_similar_paragraph(self, paragraph, number=5):
        """
        Return words most similar to a given document.
        """
        paragraph_vector = self.transform(paragraph)
        return self.glove._similarity_query(paragraph_vector, number=number)

