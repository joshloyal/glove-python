import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin, TfidfVectorizer

from .glove import Glove, check_random_state

PRETRAINED_DICT = {
    50: 'glove.6B.50d.txt',
    100: 'glove.6B.100d.txt',
    200: 'glove.6B.200d.txt',
    300: 'glove.6B.300d.txt',
}

class GloveVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word', vocabulary=None,
                 n_components=100, pre_trained=True,
                 use_idf=False, random_state=None):
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

        self.n_components = n_components
        self.pre_trained = pre_trained
        self.use_idf = use_idf
        self.random_state = random_state
        self.glove = None

    @property
    def vocabulary_(self):
        if self.glove is None:
            raise ValueError('Model not fitted.')
        return self.glove.dictionary

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents, y=None):

        if self.pre_trained:
            # get the pre-trained glove vectors
            if self.n_components in PRETRAINED_DICT:
                pretrained_vectors = PRETRAINED_DICT[self.n_components]
            else:
                raise ValueError('No pretrained glove representation with '
                                 'n_components = {}'.format(self.n_components))
            self.glove = Glove.load_stanford(pretrained_vectors)
            self.fixed_vocabulary_ = True
        else:
            raise NotImplementedError('training not implemented')

        random_state = check_random_state(self.random_state)
        self.glove.random_state = random_state
        transformed_X = self._apply_glove(raw_documents)

        # train a tfidf matrix
        if self.use_idf:
            self.tfidf = TfidfVectorizer()
            tfidf_X = self.tfidf.fit_transform(raw_documents)
            transformed_X = sp.hstack((transformed_X, tfidf_X))

        return transformed_X

    def transform(self, raw_documents):
        if self.glove is None:
            raise RuntimeError('Model glove must be fit first.')

        transformed_X = self._apply_glove(raw_documents)
        if self.use_idf:
            tfidf_X = self.tfidf.transform(raw_documents)
            transformed_X = sp.hstack((transformed_X, tfidf_X))

        return transformed_X

    def _apply_glove(self, raw_documents):
        analyze = self.build_analyzer()
        doc_vecs = np.array(
            [self.glove.transform_paragraph(analyze(doc), ignore_missing=True)
             for doc in raw_documents])

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

