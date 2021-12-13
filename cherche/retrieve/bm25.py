__all__ = ["BM25L", "BM25Okapi", "BM25Plus"]

from rank_bm25 import BM25L as rank_bm25l
from rank_bm25 import BM25Okapi as rank_bm25okapi
from rank_bm25 import BM25Plus as rank_bm25plus

from .base import _BM25


class BM25Okapi(_BM25):
    """BM25Okapi model from [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25).

    Parameters
    ----------
    on
        Field to use to match the query to the documents.
    tokenizer
        Tokenizer to use, the default one split on spaces. This tokenizer should have a
        `tokenizer.__call__` method that returns the list of tokenized tokens.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
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

    >>> retriever = retrieve.BM25Okapi(on="article", k=3, k1=1.5, b=0.75, epsilon=0.25)

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    BM25Okapi retriever
        on: article
        documents: 3

    >>> print(retriever(q="France"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'title': 'Paris'}]


    >>> retriever.add(documents=documents)
    BM25Okapi retriever
        on: article
        documents: 6

    References
    ----------
    1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)
    2. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)

    """

    def __init__(
        self, on: str, tokenizer=None, k: int = None, k1=1.5, b=0.75, epsilon=0.25
    ) -> None:
        super().__init__(on=on, bm25=rank_bm25okapi, tokenizer=tokenizer, k=k)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

    def add(self, documents: list):
        """Add documents to the retriever.

        Parameters
        ----------
        documents
            List of documents to add to the retriever.

        """
        self.documents += documents
        self.model = self.bm25(
            [
                doc[self.on].split(" ") if self.tokenizer is None else self.tokenizer(doc[self.on])
                for doc in self.documents
            ],
            k1=self.k1,
            b=self.b,
            epsilon=self.epsilon,
        )
        return self


class BM25L(_BM25):
    """BM25L model from [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25).

    Parameters
    ----------
    on
        Field to use to match the query to the documents.
    tokenizer
        Tokenizer to use, the default one split on spaces. This tokenizer should have a
        `tokenizer.__call__` method that returns the list of tokenized tokens.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
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

    >>> retriever = retrieve.BM25L(on="article", k=3, k1=1.5, b=0.75, delta=0.5)

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    BM25L retriever
        on: article
        documents: 3

    >>> print(retriever(q="France"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'title': 'Paris'}]

    >>> retriever.add(documents=documents)
    BM25L retriever
        on: article
        documents: 6

    References
    ----------
    1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)
    2. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)

    """

    def __init__(self, on: str, tokenizer=None, k: int = None, k1=1.5, b=0.75, delta=0.5) -> None:
        super().__init__(on=on, bm25=rank_bm25l, tokenizer=tokenizer, k=k)
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.model = None

    def add(self, documents: list):
        """Add documents to the retriever.

        Parameters
        ----------
        documents
            List of documents to add to the retriever.

        """
        self.documents += documents
        self.model = self.bm25(
            [
                doc[self.on].split(" ") if self.tokenizer is None else self.tokenizer(doc[self.on])
                for doc in self.documents
            ],
            k1=self.k1,
            b=self.b,
            delta=self.delta,
        )
        return self
