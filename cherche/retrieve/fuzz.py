__all__ = ["Fuzz"]

import collections
import typing

from rapidfuzz import fuzz, process, utils

from .base import Retriever


class Fuzz(Retriever):
    """[RapidFuzz](https://github.com/maxbachmann/RapidFuzz) wrapper. Rapid fuzzy string matching
    in Python and C++ using the Levenshtein Distance.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    fuzzer
        [RapidFuzz scorer](https://maxbachmann.github.io/RapidFuzz/Usage/fuzz.html): fuzz.ratio,
        fuzz.partial_ratio, fuzz.token_set_ratio, fuzz.partial_token_set_ratio,
        fuzz.token_sort_ratio, fuzz.partial_token_sort_ratio, fuzz.token_ratio,
        fuzz.partial_token_ratio, fuzz.WRatio, fuzz.QRatio, string_metric.levenshtein,
        string_metric.normalized_levenshtein
    default_process
        Pre-processing step. If set to True, documents processed by
        [RapidFuzz default process.](https://maxbachmann.github.io/RapidFuzz/Usage/utils.html)

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from rapidfuzz import fuzz

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki", "tags": ["paris", "capital"]},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki", "tags": ["canada", "montreal"]},
    ... ]

    >>> retriever = retrieve.Fuzz(
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    k = 2,
    ...    fuzzer = fuzz.partial_ratio,
    ... )

    >>> retriever.add(documents=documents)
    Fuzz retriever
         key: id
         on: title, article
         documents: 3
         fuzzer: partial_ratio

    >>> retriever("Paris")
    [{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "Paris", "author": "Wiki", "tags": ["paris", "capital"]},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
    ...    {"id": 2, "title": "Montreal", "article": "Canada.", "author": "Wiki", "tags": ["canada", "montreal"]},
    ... ]

    >>> retriever.add(documents=documents)
    Fuzz retriever
         key: id
         on: title, article
         documents: 3
         fuzzer: partial_ratio

    >>> retriever("Paris")
    [{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

    >>> documents = [
    ...    {"id": 3, "title": "Paris", "article": "Paris", "author": "Wiki", "tags": ["paris", "capital"]},
    ...    {"id": 4, "title": "Paris", "article": "Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
    ... ]

    >>> retriever.add(documents = documents)
    Fuzz retriever
         key: id
         on: title, article
         documents: 5
         fuzzer: partial_ratio

    >>> retriever("Paris")
    [{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

    References
    ----------
    1. [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        k: int,
        fuzzer=fuzz.partial_ratio,
        default_process: bool = True,
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.fuzzer = fuzzer
        self.documents = collections.OrderedDict()
        self.index = {}
        self.default_process = default_process

    def __repr__(self):
        repr = super().__repr__()
        repr += f"\n\t fuzzer: {self.fuzzer.__name__}"
        return repr

    def add(self, documents: list) -> "Fuzz":
        """Fuzz is streaming friendly.

        Parameters
        ----------
        documents
            List of documents to add to the index.

        """
        for doc in documents:

            idx = len(self.documents)

            content = " ".join([doc.get(field, "") for field in self.on])
            if self.default_process:
                content = utils.default_process(content)

            if doc[self.key] not in self.index:
                # Add new documents
                self.documents[idx] = {
                    self.key: doc[self.key],
                    "fuzzer": content,
                }
                self.index[doc[self.key]] = idx
            else:
                # Update existing document
                self.documents[self.index[doc[self.key]]] = {
                    self.key: doc[self.key],
                    "fuzzer": content,
                }

        return self

    def __call__(self, q: str, **kwargs) -> dict:
        """Retrieve documents using Fuzz.

        Parameters
        ----------
        q
            Input query.

        """
        return [
            {self.key: self.documents[idx][self.key], "similarity": float(similarity)}
            for _, similarity, idx in process.extract(
                q,
                [doc["fuzzer"] for doc in self.documents.values()],
                scorer=self.fuzzer,
                limit=self.k,
            )
        ]
