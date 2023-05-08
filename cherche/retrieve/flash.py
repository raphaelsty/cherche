__all__ = ["Flash"]

import collections
import typing
from itertools import chain

from flashtext import KeywordProcessor

from ..utils import yield_batch_single
from .base import Retriever


class Flash(Retriever):
    """FlashText Retriever. Flash aims to find documents that contain keywords such as a list of
    tags for example.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    keywords
        Keywords extractor from [FlashText](https://github.com/vi3k6i5/flashtext). If set to None,
        a default one is created.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...     {"id": 0, "title": "paris", "article": "eiffel tower"},
    ...     {"id": 1, "title": "paris", "article": "paris"},
    ...     {"id": 2, "title": "montreal", "article": "montreal is in canada"},
    ... ]

    >>> retriever = retrieve.Flash(key="id", on=["title", "article"])

    >>> retriever.add(documents=documents)
    Flash retriever
        key      : id
        on       : title, article
        documents: 4

    >>> print(retriever(q="paris", k=2))
    [{'id': 1, 'similarity': 0.6666666666666666},
     {'id': 0, 'similarity': 0.3333333333333333}]

    [{'id': 0, 'similarity': 1}, {'id': 1, 'similarity': 1}]

    >>> print(retriever(q=["paris", "montreal"]))
    [[{'id': 1, 'similarity': 0.6666666666666666},
      {'id': 0, 'similarity': 0.3333333333333333}],
     [{'id': 2, 'similarity': 1.0}]]

    References
    ----------
    1. [FlashText](https://github.com/vi3k6i5/flashtext)
    2. [Replace or Retrieve Keywords In Documents at Scale](https://arxiv.org/abs/1711.00046)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        keywords: KeywordProcessor = None,
        lowercase: bool = True,
        k: typing.Optional[int] = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k, batch_size=1)
        self.documents = collections.defaultdict(list)
        self.keywords = KeywordProcessor() if keywords is None else keywords
        self.lowercase = lowercase

    def add(self, documents: typing.List[typing.Dict[str, str]], **kwargs) -> "Flash":
        """Add keywords to the retriever.

        Parameters
        ----------
        documents
            List of documents to add to the retriever.

        """
        for document in documents:
            for field in self.on:
                if field not in document:
                    continue

                if isinstance(document[field], str):
                    words = document[field]
                    if self.lowercase:
                        words = words.lower()
                    self.documents[words].append({self.key: document[self.key]})
                    self.keywords.add_keyword(words)

                elif isinstance(document[field], list):
                    words = document[field]
                    if self.lowercase:
                        words = [word.lower() for word in words]

                    for word in words:
                        self.documents[word].append({self.key: document[self.key]})
                    self.keywords.add_keywords_from_list(words)

        return self

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int] = None,
        **kwargs,
    ) -> list:
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

        for batch in yield_batch_single(q, desc=f"{self.__class__.__name__} retriever"):
            if self.lowercase:
                batch = batch.lower()

            match = list(
                chain.from_iterable(
                    [
                        self.documents[tag]
                        for tag in self.keywords.extract_keywords(batch)
                    ]
                )
            )

            scores = collections.defaultdict(int)
            for document in match:
                scores[document[self.key]] += 1

            total = len(match)

            documents = [
                {self.key: key, "similarity": scores[key] / total}
                for key in sorted(scores, key=scores.get, reverse=True)
            ]

            documents = documents[:k] if k is not None else documents
            rank.append(documents)

        return rank[0] if isinstance(q, str) else rank
