__all__ = ["Lunr"]

import typing

from lunr import lunr

from .base import Retriever


class Lunr(Retriever):
    """Lunr is a Python implementation of Lunr.js by Oliver Nightingale. Lunr is a retriever
    dedicated for small and middle size corpus.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    documents
        Documents in Lunr retriever are static. The retriever must be reseted to index new
        documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.Lunr(key="id", on=["title", "article"], documents=documents, k=3)

    >>> retriever
    Lunr retriever
         key: id
         on: title, article
         documents: 3

    >>> print(retriever(q="paris"))
    [{'id': 0, 'similarity': 0.524}, {'id': 1, 'similarity': 0.414}]

    >>> retriever += documents

    >>> print(retriever(q="paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.524,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.414,
      'title': 'Eiffel tower'}]

    References
    ----------
    1. [Lunr.py](https://github.com/yeraydiazdiaz/lunr.py)
    2. [Lunr.js](https://lunrjs.com)
    2. [Solr](https://solr.apache.org)

    """

    def __init__(
        self, key: str, on: typing.Union[str, list], documents: list, k: int = None
    ) -> None:
        super().__init__(key=key, on=on, k=k)

        self.documents = {
            str(document[self.key]): {self.key: document[self.key]} for document in documents
        }

        self.idx = lunr(
            ref=self.key,
            fields=tuple(self.on),
            # Lunr does not handle missing fields.
            documents=[
                {field: doc.get(field, "") for field in [self.key] + self.on} for doc in documents
            ],
        )

    def __call__(self, q: str) -> list:
        """Retrieve the right document."""
        # We do not handle all Lunr possibilites right now.
        q = q.replace(":", "").replace("-", "")
        documents = [
            {**self.documents[match["ref"]], "similarity": float(match["score"])}
            for match in self.idx.search(q)
        ]
        return documents[: self.k] if self.k is not None else documents
