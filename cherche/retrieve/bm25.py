__all__ = ["BM25L", "BM25Okapi"]

import typing

from rank_bm25 import BM25L as rank_bm25l
from rank_bm25 import BM25Okapi as rank_bm25okapi

from .base import _BM25


class BM25Okapi(_BM25):
    """BM25Okapi model from [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25).

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in BM25Okapi retriever are static. The retriever must be reset to index new
        documents.
    tokenizer
        Tokenizer to use, the default one split on spaces. This tokenizer should have a
        `tokenizer.__call__` method that returns the list of tokenized tokens.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    k1
        Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].
    b
        Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].
    epsilon
        Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.BM25Okapi(key="id", on=["title", "article"], documents=documents, k=3, k1=1.5, b=0.75, epsilon=0.25)

    >>> retriever
    BM25Okapi retriever
         key: id
         on: title, article
         documents: 3

    >>> print(retriever(q="Paris"))
    [{'id': 1, 'similarity': 0.0445}, {'id': 0, 'similarity': 0.0445}]

    >>> retriever += documents

    >>> print(retriever(q="Paris"))
    [{'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.0445,
      'title': 'Eiffel tower'},
     {'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.0445,
      'title': 'Paris'}]

    References
    ----------
    1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)
    2. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: list,
        tokenizer=None,
        k: int = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        super().__init__(
            key=key, on=on, documents=documents, bm25=rank_bm25okapi, tokenizer=tokenizer, k=k
        )

        self.model = self.bm25(
            self._process_documents(documents=documents),
            k1=k1,
            b=b,
            epsilon=epsilon,
        )


class BM25L(_BM25):
    """BM25L model from [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25).

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    tokenizer
        Tokenizer to use, the default one split on spaces. This tokenizer should have a
        `tokenizer.__call__` method that returns the list of tokenized tokens.
    documents
        Documents in BM25L retriever are static. The retriever must be reseted to index new
        documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    k1
        Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].
    b
        Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].
    delta
        Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.BM25L(key="id", on=["title", "article"], documents=documents, k=3, k1=1.5, b=0.75, delta=0.5)

    >>> retriever
    BM25L retriever
        key: id
        on: title, article
        documents: 3

    >>> print(retriever(q="Paris"))
    [{'id': 1, 'similarity': 0.5679}, {'id': 0, 'similarity': 0.5679}]

    >>> retriever = retriever + documents

    >>> print(retriever(q="Paris"))
    [{'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.5679,
      'title': 'Eiffel tower'},
     {'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.5679,
      'title': 'Paris'}]

    References
    ----------
    1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)
    2. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: list,
        tokenizer=None,
        k: int = None,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ) -> None:
        super().__init__(
            key=key, on=on, documents=documents, bm25=rank_bm25l, tokenizer=tokenizer, k=k
        )

        self.model = self.bm25(
            self._process_documents(documents=documents), k1=k1, b=b, delta=delta
        )
