__all__ = ["Lunr"]

import re
import typing

from lunr import lunr

from ..utils import yield_batch_single
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

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 1, "title": "Paris", "article": "Paris is in France."},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retrieve.Lunr(
    ...     key="id",
    ...     on=["title", "article"],
    ...     documents=documents,
    ... )

    >>> retriever
    Lunr retriever
        key      : id
        on       : title, article
        documents: 3

    >>> print(retriever(q="paris", k=2))
    [{'id': 1, 'similarity': 0.268}, {'id': 0, 'similarity': 0.134}]

    >>> print(retriever(q=["paris", "montreal"], k=2))
    [[{'id': 1, 'similarity': 0.268}, {'id': 0, 'similarity': 0.134}],
     [{'id': 2, 'similarity': 0.94}]]


    References
    ----------
    1. [Lunr.py](https://github.com/yeraydiazdiaz/lunr.py)
    2. [Lunr.js](https://lunrjs.com)
    2. [Solr](https://solr.apache.org)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: list,
        k: typing.Optional[int] = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k, batch_size=1)

        self.documents = {
            str(document[self.key]): {self.key: document[self.key]}
            for document in documents
        }

        self.idx = lunr(
            ref=self.key,
            fields=tuple(self.on),
            documents=[
                {field: document.get(field, "") for field in [self.key] + self.on}
                for document in documents
            ],
        )

    def __call__(
        self,
        q: typing.Union[str, typing.List[str]],
        k: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Retrieve documents from the index.

        Parameters
        ----------
        q
            Either a single query or a list of queries.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that match the
            query will be retrieved.
        """
        rank = []

        for batch in yield_batch_single(
            array=q, desc=f"{self.__class__.__name__} retriever"
        ):
            batch = re.sub("[^a-zA-Z0-9 \n\.]", " ", batch)
            documents = [
                {**self.documents[match["ref"]], "similarity": match["score"]}
                for match in self.idx.search(batch)
            ]
            documents = documents[:k] if k is not None else documents
            rank.append(documents)

        return rank[0] if isinstance(q, str) else rank
