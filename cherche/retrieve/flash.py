__all__ = ["Flash"]

import collections
from itertools import chain
from typing import Union

from flashtext import KeywordProcessor

from .base import Retriever


class Flash(Retriever):
    """FlashText Retriever. Flash aims to find documents that contain keywords such as a list of
    tags for example.

    Parameters
    ----------
    on
        Field to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.
    keywords
        Keywords extractor from [FlashText](https://github.com/vi3k6i5/flashtext). If set to None,
        a default one is created.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> retriever = retrieve.Flash(on="tags", k=2)

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki", "tags": ["paris", "capital"]},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki", "tags": ["canada", "montreal"]},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> print(retriever(q="paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'tags': ['paris', 'capital'],
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'tags': ['paris', 'eiffel', 'tower'],
      'title': 'Eiffel tower'}]

    References
    ----------
    1. [FlashText](https://github.com/vi3k6i5/flashtext)
    2. [Replace or Retrieve Keywords In Documents at Scale](https://arxiv.org/abs/1711.00046)

    """

    def __init__(self, on: str, k: int = None, keywords=None) -> None:
        super().__init__(on=on, k=k)
        self.documents = collections.defaultdict(list)
        self.keywords = KeywordProcessor() if keywords is None else keywords

    def add(self, documents: list):
        """Add keywords to the retriever.

        Parameters
        ----------
        documents
            List of documents to add to the retriever.

        """
        for document in documents:
            if isinstance(document[self.on], list):
                for tag in document[self.on]:
                    self.documents[tag].append(document)
            else:
                self.documents[document[self.on]].append(document)
            self._add(document=document[self.on])
        return self

    def _add(self, document: Union[list, str]):
        """Update keywords using dict, list or string."""
        if isinstance(document, list):
            self.keywords.add_keywords_from_list(document)
        elif isinstance(document, str):
            self.keywords.add_keyword(document)
        return self

    def __call__(self, q: str) -> list:
        """Retrieve tagss."""
        documents = list(
            chain.from_iterable([self.documents[tag] for tag in self.keywords.extract_keywords(q)])
        )

        # Remove duplicates documents
        documents = [i for n, i in enumerate(documents) if i not in documents[n + 1 :]]
        return documents[: self.k] if self.k is not None else documents
