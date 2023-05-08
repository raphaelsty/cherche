__all__ = ["TfIdf"]

import typing

import numpy as np
import tqdm
from scipy.sparse import csc_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils import yield_batch
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
    ...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 1, "title": "Paris", "article": "Paris is in France."},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

    >>> retriever
    TfIdf retriever
        key      : id
        on       : title, article
        documents: 3

    >>> print(retriever(q=["paris", "montreal paris"]))
    [[{'id': 1, 'similarity': 0.366173437788525},
      {'id': 0, 'similarity': 0.23008513690129015}],
     [{'id': 2, 'similarity': 0.6568592005036291},
      {'id': 1, 'similarity': 0.18870017418263602},
      {'id': 0, 'similarity': 0.07522017339345569}]]

    >>> print(retriever(["unknown", "montreal paris"], k=2))
    [[],
     [{'id': 2, 'similarity': 0.6568592005036291},
      {'id': 1, 'similarity': 0.18870017418263602}]]

    >>> print(retriever(q="paris", k=2))
    [{'id': 1, 'similarity': 0.366173437788525},
     {'id': 0, 'similarity': 0.23008513690129015}]

    References
    ----------
    1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: typing.List[typing.Dict[str, str]],
        tfidf: TfidfVectorizer = None,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(key=key, on=on, k=k, batch_size=batch_size)

        self.tfidf = (
            TfidfVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char")
            if tfidf is None
            else tfidf
        )

        self.documents = [{self.key: document[self.key]} for document in documents]

        self.matrix = csc_matrix(
            self.tfidf.fit_transform(
                [
                    " ".join([doc.get(field, "") for field in self.on])
                    for doc in documents
                ]
            )
        )

        self.k = len(self.documents) if k is None else k
        self.n = len(self.documents)

    def top_k_by_partition(
        self, similarities: np.ndarray, k: int
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Top k elements by partition."""
        similarities *= -1

        if k < self.n:
            ind = np.argpartition(similarities, k, axis=-1)

            # k non-sorted indices
            ind = np.take(ind, np.arange(k), axis=-1)

            # k non-sorted values
            similarities = np.take_along_axis(similarities, ind, axis=-1)

            # sort within k elements
            ind_part = np.argsort(similarities, axis=-1)
            ind = np.take_along_axis(ind, ind_part, axis=-1)

        else:
            ind_part = np.argsort(similarities, axis=-1)
            ind = ind_part

        similarities *= -1
        val = np.take_along_axis(similarities, ind_part, axis=-1)
        return ind, val

    def __call__(
        self,
        q: typing.Union[str, typing.List[str]],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Retrieve documents from batch of queries.

        Parameters
        ----------
        q
            Either a single query or a list of queries.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that match the
            query will be retrieved.
        batch_size
            Batch size to use to retrieve documents.
        """
        k = k if k is not None else self.k

        ranked = []

        for batch in yield_batch(
            q,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            desc=f"{self.__class__.__name__} retriever",
        ):
            similarities = self.tfidf.transform(batch).dot(self.matrix.T).toarray()

            batch_match, batch_similarities = self.top_k_by_partition(
                similarities=similarities, k=k
            )

            for match, similarities in zip(batch_match, batch_similarities):
                ranked.append(
                    [
                        {**self.documents[idx], "similarity": similarity}
                        for idx, similarity in zip(match, similarities)
                        if similarity > 0
                    ]
                )

        return ranked[0] if isinstance(q, str) else ranked
