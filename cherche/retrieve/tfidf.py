__all__ = ["TfIdf"]

import typing

import numpy as np
from scipy.sparse import csc_matrix, hstack
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
    >>> from sklearn.feature_extraction.text import TfidfVectorizer


    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 1, "title": "Montreal", "article": "Montreal is in Canada."},
    ...     {"id": 2, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 3, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retrieve.TfIdf(
    ...     key="id",
    ...     on=["title", "article"],
    ...     documents=documents,
    ... )

    >>> documents = [
    ...     {"id": 4, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 5, "title": "Montreal", "article": "Montreal is in Canada."},
    ...     {"id": 6, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 7, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retriever.add(documents)

    >>> print(retriever(q=["paris", "canada"], k=4))
    [[{'id': 6, 'similarity': 0.5404109029445249},
      {'id': 0, 'similarity': 0.5404109029445249},
      {'id': 2, 'similarity': 0.5404109029445249},
      {'id': 4, 'similarity': 0.5404109029445249}],
     [{'id': 7, 'similarity': 0.3157669764669935},
      {'id': 5, 'similarity': 0.3157669764669935},
      {'id': 3, 'similarity': 0.3157669764669935},
      {'id': 1, 'similarity': 0.3157669764669935}]]

    >>> print(retriever(["unknown", "montreal paris"], k=2))
    [[],
     [{'id': 7, 'similarity': 0.7391866872635209},
      {'id': 5, 'similarity': 0.7391866872635209}]]


    >>> print(retriever(q="paris"))
    [{'id': 6, 'similarity': 0.5404109029445249},
     {'id': 0, 'similarity': 0.5404109029445249},
     {'id': 2, 'similarity': 0.5404109029445249},
     {'id': 4, 'similarity': 0.5404109029445249}]

    References
    ----------
    1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
    2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: typing.List[typing.Dict[str, str]] = None,
        tfidf: TfidfVectorizer = None,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
        fit: bool = True,
    ) -> None:
        super().__init__(key=key, on=on, k=k, batch_size=batch_size)

        self.tfidf = (
            TfidfVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char_wb")
            if tfidf is None
            else tfidf
        )

        self.documents = [{self.key: document[self.key]} for document in documents]
        self.duplicates = {document[self.key]: True for document in documents}

        method = self.tfidf.fit_transform if fit else self.tfidf.transform

        self.matrix = csc_matrix(
            method(
                [
                    " ".join([doc.get(field, "") for field in self.on])
                    for doc in documents
                ]
            ),
            dtype=np.float32,
        ).T

        self.k = len(self.documents) if k is None else k
        self.n = len(self.documents)

    def add(
        self,
        documents: list,
        batch_size: int = 100_000,
        tqdm_bar: bool = False,
        **kwargs,
    ):
        """Add new documents to the TFIDF retriever. The tfidf won't be refitted."""
        for batch in yield_batch(
            documents,
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
            desc="Adding documents to TfIdf retriever",
        ):
            batch = [
                document
                for document in batch
                if document[self.key] not in self.duplicates
            ]

            if not batch:
                continue

            sparse_matrix = csc_matrix(
                self.tfidf.transform(
                    [
                        " ".join([doc.get(field, "") for field in self.on])
                        for doc in batch
                    ]
                ),
                dtype=np.float32,
            ).T

            self.matrix = hstack((self.matrix, sparse_matrix))

            for document in batch:
                self.documents.append({self.key: document[self.key]})
                self.duplicates[document[self.key]] = True

            self.n += len(batch)

        return self

    def top_k(self, similarities: csc_matrix, k: int):
        """Return the top k documents for each query."""
        matchs, scores = [], []
        for row in similarities:
            _k = min(row.data.shape[0] - 1, k)
            ind = np.argpartition(row.data, kth=_k, axis=0)[:k]
            similarity = np.take_along_axis(row.data, ind, axis=0)
            indices = np.take_along_axis(row.indices, ind, axis=0)
            ind = np.argsort(similarity, axis=0)
            scores.append(-1 * np.take_along_axis(similarity, ind, axis=0))
            matchs.append(np.take_along_axis(indices, ind, axis=0))
        return matchs, scores

    def __call__(
        self,
        q: typing.Union[str, typing.List[str]],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        tqdm_bar: bool = True,
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
            tqdm_bar=tqdm_bar,
        ):
            similarities = -1 * self.tfidf.transform(batch).dot(self.matrix)

            batch_match, batch_similarities = self.top_k(similarities=similarities, k=k)

            for match, similarities in zip(batch_match, batch_similarities):
                ranked.append(
                    [
                        {**self.documents[idx], "similarity": similarity}
                        for idx, similarity in zip(match, similarities)
                        if similarity > 0
                    ]
                )

        return ranked[0] if isinstance(q, str) else ranked
