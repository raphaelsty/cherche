__all__ = ["Lunr"]

import copy
import typing

from lunr import lunr

from .base import Retriever


class Lunr(Retriever):
    """Lunr is a Python implementation of Lunr.js by Oliver Nightingale. Lunr is a retriever
    dedicated for small and middle size corpus.

    Parameters
    ----------
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> retriever = retrieve.Lunr(on=["title", "article"], k=3)

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    Lunr retriever
         on: title, article
         documents: 3

    >>> print(retriever(q="paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'title': 'Eiffel tower'}]

    References
    ----------
    1. (Lunr.py)[https://github.com/yeraydiazdiaz/lunr.py]

    """

    def __init__(self, on: typing.Union[str, list], k: int = None) -> None:
        super().__init__(on=on, k=k)
        self.idx = None

    def add(self, documents: list) -> "Lunr":
        self.documents += documents
        self.documents = [{**document, "ref": idx} for idx, document in enumerate(self.documents)]
        self.idx = lunr(ref="ref", fields=tuple(self.on), documents=self.documents)
        return self

    def __call__(self, q: str) -> list:
        """Retrieve the right document."""
        documents = []
        for match in self.idx.search(q):
            document = copy.copy(self.documents[int(match["ref"])])
            document.pop("ref")
            documents.append(document)

        return documents[: self.k] if self.k is not None else documents
