__all__ = ["Flash"]

import collections
from itertools import chain
from typing import Union

from flashtext import KeywordProcessor

from .base import Retriever


class Flash(Retriever):
    """FlashText Retriever.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> retriever = retrieve.Flash(on="tag", k=2)

    >>> documents = [
    ...     {"url": "ckb/github.com", "tag": "Transformers", "date": "10-11-2021", "label": "Transformers are heavy."},
    ...     {"url": "mkb/github.com", "tag": ["Transformers", "Pytorch"], "date": "22-11-2021", "label": "Transformers with Pytorch"},
    ...     {"url": "blp/github.com", "tag": "Github", "date": "22-11-2020", "label": "Github is a great tool."},
    ... ]

    >>> retriever = retriever.add(documents=documents)

    >>> print(retriever(q="Transformers with Pytorch"))
    [{'date': '10-11-2021',
      'label': 'Transformers are heavy.',
      'tag': 'Transformers',
      'url': 'ckb/github.com'},
     {'date': '22-11-2021',
      'label': 'Transformers with Pytorch',
      'tag': ['Transformers', 'Pytorch'],
      'url': 'mkb/github.com'}]

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
