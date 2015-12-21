# _*_ coding: utf-8 _*_
from glove import GloveVectorizer
from sklearn.datasets import fetch_20newsgroups

def test_smoke():
    categories = [
        'alt.atheism',
        'soc.religion.christian',
        'comp.graphics',
        'sci.med'
    ]
    twenty_train = fetch_20newsgroups(
        subset='train', categories=categories, shuffle=True, random_state=42)

    vec = GloveVectorizer(n_components=100, random_state=42, verbose=True)
    X = vec.fit_transform(twenty_train.data)
    assert X.shape[1] == 100
