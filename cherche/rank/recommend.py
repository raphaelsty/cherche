from __future__ import annotations

import typing

import numpy as np

from ..index import Milvus
from ..similarity import cosine
from .base import MemoryStore, Ranker

__all__ = ["Recommend"]


class Recommend(Ranker):
    """Collaborative filtering as a ranker. Recommend is compatible with the library [Implicit](https://github.com/benfred/implicit).

    Parameters
    ----------
    key
        Field identifier of each document.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.
    index
        Index that will store the embeddings of documents and perform the similarity search.
        The default index is Faiss. We can choose index.Milvus also.
    store
        Index that will store the embeddings of users. By default, it store users embeddings in
        memory. We can choose index.Milvus also.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import rank, utils
    >>> from implicit.nearest_neighbours import bm25_weight
    >>> from implicit.als import AlternatingLeastSquares

    >>> documents = [
    ...    {"id": "a", "title": "Paris", "author": "Paris"},
    ...    {"id": "b", "title": "Madrid", "author": "Madrid"},
    ...    {"id": "c", "title": "Montreal", "author": "Montreal"},
    ... ]

    >>> ratings = {
    ...    "Max": {"a": 1, "c": 1},
    ...    "Adil": {"b": 1, "d": 2},
    ...    "Robin": {"b": 1, "d": 1},
    ...    "Geoffrey": {"a": 1, "c": 1},
    ... }

    >>> index_users, index_documents, sparse_ratings = utils.users_items_sparse(ratings=ratings)

    >>> model = AlternatingLeastSquares(
    ...     factors=64,
    ...     regularization=0.05,
    ...     alpha=2.0,
    ...     iterations=100,
    ...     random_state=42,
    ... )

    >>> model.fit(sparse_ratings)

    >>> embeddings_users = {
    ...    user: embedding for user, embedding in zip(index_users, model.user_factors)
    ... }

    >>> embeddings_documents = {
    ...    document: embedding
    ...    for document, embedding in zip(index_documents, model.item_factors)
    ... }

    >>> recommend = rank.Recommend(
    ...    key="id",
    ...    k = 10,
    ... )

    >>> recommend.add(
    ...    documents=documents,
    ...    embeddings_documents=embeddings_documents,
    ...    embeddings_users=embeddings_users,
    ... )
    Recommend ranker
        key: id
        Users: 4
        Documents: 3

    >>> print(recommend(user="Geoffrey", documents=documents))
    [{'author': 'Paris',
      'id': 'a',
      'similarity': 1.0000001192092896,
      'title': 'Paris'},
     {'author': 'Montreal',
      'id': 'c',
      'similarity': 0.9999998807907104,
      'title': 'Montreal'},
     {'author': 'Madrid',
      'id': 'b',
      'similarity': 4.273452987035853e-07,
      'title': 'Madrid'}]

    >>> recommend(user="Geoffrey", documents=[{"id": "unknown", "title": "unknown", "author": "unknown"}])
    [{'id': 'unknown', 'title': 'unknown', 'author': 'unknown', 'similarity': 0}]

    >>> print(recommend(user="unknown", documents=documents))
    [{'author': 'Paris', 'id': 'a', 'similarity': 0, 'title': 'Paris'},
     {'author': 'Madrid', 'id': 'b', 'similarity': 0, 'title': 'Madrid'},
     {'author': 'Montreal', 'id': 'c', 'similarity': 0, 'title': 'Montreal'}]

    References
    ----------
    1. [Implicit](https://github.com/benfred/implicit)
    2. [Implicit documentation](https://benfred.github.io/implicit/)
    3. [Logistic Matrix Factorization for Implicit Feedback Data](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)
    4. [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618.pdf)
    5. [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)

    """

    def __init__(
        self,
        key: str,
        k: int = None,
        similarity=cosine,
        store_items=MemoryStore(),
        store_users=MemoryStore(),
    ) -> None:
        super().__init__(
            key=key,
            on="",
            encoder=None,
            k=k,
            similarity=similarity,
            store=store_items,
        )

        self.store_users = store_users

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n \t key: {self.key}"
        repr += f"\n \t Users: {len(self.store_users)}"
        repr += f"\n \t Documents: {len(self.store)}"
        return repr

    def add(
        self,
        documents: list,
        embeddings_documents: dict,
        embeddings_users: dict,
        **kwargs,
    ) -> "Recommend":
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
        index = {document[self.key]: document for document in documents}

        self.store.add(
            documents=[
                index[key] for key in embeddings_documents.keys() if key in index
            ],
            embeddings=np.array(
                [
                    embedding
                    for key, embedding in embeddings_documents.items()
                    if key in index
                ]
            ),
            key=self.key,
        )

        self.store_users.add(
            users=[key for key in embeddings_users.keys()],
            embeddings=[embedding for embedding in embeddings_users.values()],
        )
        return self

    def __call__(
        self,
        user: typing.Union[str, int],
        documents: list,
        expr: str = None,
        consistency_level: str = None,
        partition_names: list = None,
        **kwargs,
    ) -> list:
        """Retrieve documents from user id.

        Parameters
        ----------
        user
            User id.
        documents
            List of documents to rank.
        """
        known_user, embedding_user, unknown_user = self.store_users.get(values=[user])

        # Unknown user.
        if not known_user:
            return [{**document, "similarity": 0} for document in documents]

        known, embeddings, unknown = self.store.get(
            **{
                "key": self.key,
                "values": [document[self.key] for document in documents],
            }
        )

        # Unknown documents.
        if not known:
            return [{**document, "similarity": 0} for document in documents]

        index = {document[self.key]: document for document in documents}
        index_known = {i: key  for i, key in enumerate(known)}

        ranked = [
            {**index[index_known[key]], "similarity": similarity}
            for key, similarity in self.similarity(
                emb_q=embedding_user[0], emb_documents=embeddings
            )
        ]

        # Addind unknown documents
        ranked += [{**index[key], "similarity": 0} for key in unknown]
        return ranked[:self.k] if self.k is not None else ranked
