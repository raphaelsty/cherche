__all__ = ["TfIdf"]


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .base import Retriever


class TfIdf(Retriever):
    """TfIdf retriever based on cosine similarities.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> retriever = retrieve.TfIdf(on="title")

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    TfIdf retriever
         on: title
         documents: 3

    >>> print(retriever(q="Transformers", k=2))
    [{'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers.',
      'url': 'ckb/github.com'},
     {'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers.',
      'url': 'blp/github.com'}]

    References
    ----------
    1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(self, on: str, tfidf: TfidfVectorizer = None) -> None:
        super().__init__(on=on)
        self.tfidf = TfidfVectorizer() if tfidf is None else tfidf
        self.matrix = None

    def add(self, documents: list):
        """Add documents."""
        self.documents += documents
        self.matrix = self.tfidf.fit_transform([doc[self.on] for doc in self.documents])
        return self

    def __call__(self, q: list, k: int = None) -> list:
        """Retrieve the right document."""
        similarities = linear_kernel(self.tfidf.transform([q]), self.matrix).flatten()
        documents = [self.documents[index] for index in (-similarities).argsort()]
        return documents[:k] if k is not None else documents
