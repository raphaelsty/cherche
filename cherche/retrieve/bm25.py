__all__ = ["BM25"]

from rank_bm25 import BM25Okapi

from .base import Retriever


class BM25(Retriever):
    """BM25 Retriever.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> retriever = retrieve.BM25(on="title")

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    BM25 retriever
        on: title
        documents: 3

    >>> print(retriever(q="Transformers", k=2))
    [{'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'},
     {'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'}]

    References
    ----------
    1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)

    """

    def __init__(self, on: str, tokenizer=None) -> None:
        super().__init__(on)
        self.bm25 = None
        self.tokenizer = tokenizer

    def add(self, documents: list):
        """Add documents."""
        self.documents += documents
        self.bm25 = BM25Okapi(
            [
                doc[self.on].split(" ") if self.tokenizer is None else self.tokenizer(doc[self.on])
                for doc in self.documents
            ]
        )
        return self

    def __call__(self, q: str, k: int = None) -> list:
        """Retrieve the right document."""
        q = q.split(" ") if self.tokenizer is None else self.tokenizer(q)
        similarities = self.bm25.get_scores(q)
        documents = [self.documents[index] for index in similarities.argsort()]
        return documents[:k] if k is not None else documents
