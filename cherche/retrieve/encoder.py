__all__ = ["Encoder"]

import typing

import numpy as np

from .base import BaseEncoder


class Encoder(BaseEncoder):
    """Encoder as a retriever using Faiss Index.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Field to use to retrieve documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    k = 2,
    ... )

    >>> retriever.add(documents)
    Encoder retriever
         key: id
         on: title, article
         documents: 3

    >>> print(retriever("Paris"))
    [{'id': 0, 'similarity': 1.47281}, {'id': 1, 'similarity': 1.02935}]

    >>> documents = [
    ...    {"id": 3, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 4, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 5, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever.add(documents)
    Encoder retriever
         key: id
         on: title, article
         documents: 6

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ...    {"id": 3, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 4, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 5, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever += documents

    >>> print(retriever("Paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 3,
      'similarity': 1.47281,
      'title': 'Paris'},
     {'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 1.47281,
      'title': 'Paris'}]

    References
    ----------
    1. [Faiss](https://github.com/facebookresearch/faiss)

    """

    def __init__(
        self, encoder, key: str, on: typing.Union[str, list], k: int, path: str = None
    ) -> None:
        super().__init__(encoder=encoder, key=key, on=on, k=k, path=path)

    def __call__(self, q: str) -> list:
        distances, indexes = self.tree.search(
            np.array(
                [self.encoder(q) if q not in self.q_embeddings else self.q_embeddings[q]]
            ).astype(np.float32),
            self.k if self.k is not None else len(self.documents),
        )
        ranked = []
        for index, distance in zip(indexes[0], distances[0]):
            document = self.documents[index]
            document["similarity"] = float(1 / distance) if distance > 0 else 0.0
            ranked.append(document)
        return ranked
