__all__ = ["Fuzz"]

import collections
import typing

from rapidfuzz import fuzz, process, utils

from ..utils import yield_batch_single
from .base import Retriever


class Fuzz(Retriever):
    """[RapidFuzz](https://github.com/maxbachmann/RapidFuzz) wrapper. Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
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
    ...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
    ...     {"id": 1, "title": "Paris", "article": "Paris is in France."},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada."},
    ... ]

    >>> retriever = retrieve.Fuzz(
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    fuzzer = fuzz.partial_ratio,
    ... )

    >>> retriever.add(documents=documents)
    Fuzz retriever
        key      : id
        on       : title, article
        documents: 3

    >>> print(retriever(q="paris", k=2))
    [{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

    >>> print(retriever(q=["paris", "montreal"], k=2))
    [[{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}],
     [{'id': 2, 'similarity': 100.0}, {'id': 1, 'similarity': 37.5}]]

    >>> print(retriever(q=["unknown", "montreal"], k=2))
    [[{'id': 2, 'similarity': 40.0}, {'id': 0, 'similarity': 36.36363636363637}],
     [{'id': 2, 'similarity': 100.0}, {'id': 1, 'similarity': 37.5}]]

    References
    ----------
    1. [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        fuzzer=fuzz.partial_ratio,
        default_process: bool = True,
        k: typing.Optional[int] = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k, batch_size=1)
        self.fuzzer = fuzzer
        self.documents = collections.OrderedDict()
        self.index = {}
        self.default_process = default_process

    def add(self, documents: typing.List[typing.Dict[str, str]], **kwargs) -> "Fuzz":
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

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int] = None,
        **kwargs,
    ) -> dict:
        """Retrieve documents from the index.

        Parameters
        ----------
        q
            Either a single query or a list of queries.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that match the
            query will be retrieved.
        """
        if k is None:
            k = len(self.documents)

        rank = []

        for batch in yield_batch_single(
            array=q, desc=f"{self.__class__.__name__} retriever"
        ):
            rank.append(
                [
                    {self.key: self.documents[idx][self.key], "similarity": similarity}
                    for _, similarity, idx in process.extract(
                        batch,
                        [doc["fuzzer"] for doc in self.documents.values()],
                        scorer=self.fuzzer,
                        limit=k,
                    )
                ]
            )

        return rank[0] if isinstance(q, str) else rank
