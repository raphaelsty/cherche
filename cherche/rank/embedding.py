from __future__ import annotations

import typing

import numpy as np

from .base import MemoryStore, Ranker

__all__ = ["Embedding"]


class Embedding(Ranker):
    """Collaborative filtering as a ranker. Recommend is compatible with the library [Implicit](https://github.com/benfred/implicit).

    Parameters
    ----------
    key
        Field identifier of each document.
    normalize
        If set to True, the similarity measure is cosine similarity, if set to False, similarity
        measure is dot product.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": "a", "title": "Paris"},
    ...    {"id": "b", "title": "Madrid"},
    ...    {"id": "c", "title": "Montreal"},
    ... ]

    >>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    >>> embeddings_documents = encoder.encode([
    ...    document["title"] for document in documents
    ... ])

    >>> recommend = rank.Embedding(
    ...    key="id",
    ... )

    >>> recommend.add(
    ...    documents=documents,
    ...    embeddings_documents=embeddings_documents,
    ... )
    Embedding ranker
        key      : id
        documents: 3
        normalize: True

    >>> match = recommend(
    ...     q=encoder.encode("Paris"),
    ...     documents=documents,
    ...     k=2
    ... )

    >>> print(match)
    [{'id': 'a', 'similarity': 1.0, 'title': 'Paris'},
     {'id': 'c', 'similarity': 0.57165134, 'title': 'Montreal'}]

    >>> queries = [
    ...    "Paris",
    ...    "Madrid",
    ...    "Montreal"
    ... ]

    >>> match = recommend(
    ...     q=encoder.encode(queries),
    ...     documents=[documents] * 3,
    ...     k=2
    ... )

    >>> print(match)
    [[{'id': 'a', 'similarity': 1.0, 'title': 'Paris'},
      {'id': 'c', 'similarity': 0.57165134, 'title': 'Montreal'}],
     [{'id': 'b', 'similarity': 1.0, 'title': 'Madrid'},
      {'id': 'a', 'similarity': 0.49815434, 'title': 'Paris'}],
     [{'id': 'c', 'similarity': 0.9999999, 'title': 'Montreal'},
      {'id': 'a', 'similarity': 0.5716514, 'title': 'Paris'}]]

    """

    def __init__(
        self,
        key: str,
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(
            key=key,
            on="",
            encoder=None,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n\tkey      : {self.key}"
        repr += f"\n\tdocuments: {len(self)}"
        repr += f"\n\tnormalize: {self.normalize}"
        return repr

    def add(
        self,
        documents: list,
        embeddings_documents: typing.List[np.ndarray],
        **kwargs,
    ) -> "Embedding":
        """Add embeddings both documents and users.

        Parameters
        ----------
        documents
            List of documents.
        embeddings_documents
            Embeddings of the documents ordered as the list of documents.
        users
            List of users.
        embeddings_users
            Embeddings of the users ordered as the list of users.

        """
        self.store.add(
            documents=documents,
            embeddings=embeddings_documents,
            key=self.key,
        )

        return self

    def __call__(
        self,
        q: np.ndarray,
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Retrieve documents from user id.

        Parameters
        ----------
        user
            User id.
        documents
            List of documents to rank.
        """
        if k is None:
            k = self.k

        if k is None:
            k = len(self)

        documents = [documents] if len(q.shape) == 1 else documents
        known, embeddings, _ = self.store.get(documents=documents)

        ranked = self.rank(
            embeddings_queries=q,
            embeddings_documents={
                key: embedding for key, embedding in zip(known, embeddings)
            },
            documents=documents,
            k=k,
            batch_size=batch_size if batch_size is not None else self.batch_size,
        )

        return ranked[0] if len(q.shape) == 1 else ranked
