__all__ = ["TfIdf"]

import typing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .base import Retriever


class TfIdf(Retriever):
    """TfIdf retriever based on cosine similarities.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in TFIdf retriever are static. The retriever must be reseted to index new
        documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    tfidf
        TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=3)

    >>> retriever
    TfIdf retriever
         key: id
         on: title, article
         documents: 3

    >>> print(retriever(q="paris"))
    [{'id': 0}, {'id': 1}]

    >>> retriever += documents

    >>> print(retriever(q="paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'title': 'Eiffel tower'}]

    References
    ----------
    1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: list,
        k: int = None,
        tfidf: TfidfVectorizer = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.tfidf = TfidfVectorizer() if tfidf is None else tfidf

        self.documents = {
            index: {self.key: document[self.key]} for index, document in enumerate(documents)
        }

        self.matrix = self.tfidf.fit_transform(
            [" ".join([doc.get(field, "") for field in self.on]) for doc in documents]
        )

    def __call__(self, q: str) -> list:
        """Retrieve the right document."""
        similarities = linear_kernel(self.tfidf.transform([q]), self.matrix).flatten()
        documents = [
            self.documents[index] for index in (-similarities).argsort() if similarities[index] > 0
        ]
        return documents[: self.k] if self.k is not None else documents
