__all__ = ["Meilisearch"]


import typing

import more_itertools
import tqdm

from .base import Retriever


class Meilisearch(Retriever):
    """Meilisearch is a RESTful search API. It aims to be a ready-to-go solution for everyone who wants a fast and relevant search experience for their end-users.

    Parameters
    ----------
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
         will be retrieved.
    index
        Meilisearch index. Meilisearch will create the index if it does not
        exist.

    Examples
    --------

    >>> import meilisearch
    >>> from cherche import retrieve
    >>> from pprint import pprint as print

    >>> documents = [
    ...    {"key": 1, "type": "movie", "title": "Carol", "genres": ["Romance", "Drama"]},
    ...    {"key": 2, "type": "movie", "title": "Wonder Woman", "genres": ["Action", "Adventure"]},
    ...    {"key": 3, "type": "movie", "title": "Life of Pi", "genres": ["Adventure", "Drama"]}
    ... ]

    >>> client = meilisearch.Client('http://127.0.0.1:7700', 'masterKey')

    >>> retriever = retrieve.Meilisearch(
    ...    key="key", on=["type", "title", "genres"], k=20, index=client.index("movies"))

    >>> retriever.add(documents)
    Meilisearch retriever
        key: key
        on: type, title, genres
        documents: 3

    >>> print(retriever("movie"))
    [{'genres': ['Romance', 'Drama'],
      'key': 1,
      'similarity': 1.0,
      'title': 'Carol',
      'type': 'movie'},
     {'genres': ['Action', 'Adventure'],
      'key': 2,
      'similarity': 0.5,
      'title': 'Wonder Woman',
      'type': 'movie'},
     {'genres': ['Adventure', 'Drama'],
      'key': 3,
      'similarity': 0.3333333333333333,
      'title': 'Life of Pi',
      'type': 'movie'}]

    References
    ----------
    1. [meilisearch-python](https://github.com/meilisearch/meilisearch-python)
    2. [Meilisearch documentation](https://docs.meilisearch.com/learn/getting_started/quick_start.html#setup-and-installation)
    3. [Meilisearch settings](https://docs.meilisearch.com/reference/api/settings.html#settings-object)

    """

    def __init__(
        self,
        key: str,
        on: str,
        index,
        k: typing.Optional[int] = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.index = index
        self.index.update_searchable_attributes(self.on)
        self.fields = {"matchingStrategy": "last"}
        if self.k is not None:
            self.fields["limit"] = self.k

    def add(self, documents: list, batch_size=128, **kwargs) -> "Meilisearch":
        """Meilisearch is streaming friendly.

        Parameters
        ----------
        documents
            List of documents to add to the index.

        """
        for document in documents:
            document["id"] = document.pop(self.key)

        for batch in tqdm.tqdm(
            more_itertools.chunked(documents, batch_size),
            position=0,
            desc="Meilisearch indexing.",
            total=1 + len(documents) // batch_size,
        ):
            self.index.add_documents(batch)

        return self

    def __len__(self) -> int:
        return self.index.get_stats().number_of_documents

    def __call__(self, q: str, opt_params=None, **kwargs) -> list:
        """Retrieve the right document.

        Parameters
        ----------
        q
            Input query.

        """
        documents = []

        if opt_params is None:
            opt_params = self.fields

        for rank, document in enumerate(
            self.index.search(q, opt_params=opt_params)["hits"]
        ):
            document[self.key] = document.pop("id")
            document["similarity"] = 1 / (rank + 1)
            documents.append(document)
        return documents

    def reset(self) -> "Meilisearch":
        self.index.delete()
        return self
