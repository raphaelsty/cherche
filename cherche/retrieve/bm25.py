__all__ = ["BM25"]

from rank_bm25 import BM25Okapi

from .base import Retriever


class BM25(Retriever):
    """BM25 Retriever.

    Parameters
    ----------

        on: Field that BM25 will use to search relevant documents.
        bm25: Model from https://github.com/dorianbrown/rank_bm25.
        tokenizer: Default tokenizer consist by splitting on space. This tokenizer should have a
            tokenizer.__call__ method that returns the list of tokens from an input sentence.
        k: Number of documents to retrieve.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from rank_bm25 import BM25Okapi

    >>> retriever = retrieve.BM25(on="title", k=3, bm25 = BM25Okapi)

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "It is quite windy in London.", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch and Transformers .", "date": "22-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with Transformers .", "date": "22-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch and Transformers .", "date": "22-11-2021"},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    BM25 retriever
        on: title
        documents: 5

    >>> print(retriever(q="PyTorch Transformers"))
    [{'date': '22-11-2021',
      'title': 'Github Library with PyTorch and Transformers .',
      'url': 'mkb/github.com'},
     {'date': '22-11-2021',
      'title': 'Github Library with PyTorch and Transformers .',
      'url': 'mkb/github.com'},
     {'date': '22-11-2021',
      'title': 'Github Library with PyTorch .',
      'url': 'mkb/github.com'}]


    References
    ----------
    1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)

    """

    def __init__(self, on: str, bm25=None, tokenizer=None, k: int = None) -> None:
        super().__init__(on=on, k=k)
        self.bm25 = bm25 if bm25 is not None else BM25Okapi
        self.tokenizer = tokenizer

    def add(self, documents: list):
        """Add documents."""
        self.documents += documents
        self.bm25 = self.bm25(
            [
                doc[self.on].split(" ") if self.tokenizer is None else self.tokenizer(doc[self.on])
                for doc in self.documents
            ]
        )
        return self

    def __call__(self, q: str) -> list:
        """Retrieve the right document using BM25."""
        q = q.split(" ") if self.tokenizer is None else self.tokenizer(q)
        similarities = abs(self.bm25.get_scores(q))
        documents = [self.documents[index] for index in (-similarities).argsort()]
        return documents[: self.k] if self.k is not None else documents
