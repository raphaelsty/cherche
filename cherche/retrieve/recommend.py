import typing

import faiss
import numpy as np

from ..index import Faiss, Milvus
from .base import Retriever

__all__ = ["Recommend"]


class MemoryStore:
    """Store user embeddings in memory."""

    def __init__(self):
        self.embeddings = {}

    def __len__(self):
        return len(self.embeddings)

    def add(self, users: list, embeddings: np.ndarray):
        for user, embedding in zip(users, embeddings):
            self.embeddings[user] = np.array(embedding)
        return self

    def get(self, values: list, **kwargs) -> typing.Tuple[list, list, list]:
        """Get users embeddings from memory.

        Parameters
        ----------
        values
            List of user ids.

        """
        known, user_embeddings, unknown = [], [], []
        for user in values:
            embedding = self.embeddings.get(user, None)
            if embedding is not None:
                known.append(user)
                user_embeddings.append(embedding)
            else:
                unknown.append(user)
        return known, user_embeddings, unknown


class Recommend(Retriever):
    """Collaborative filtering as a retriever. Recommend is compatible with the library [Implicit](https://github.com/benfred/implicit).

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
    >>> from cherche import retrieve, utils
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

    >>> recommend = retrieve.Recommend(
    ...    key="id",
    ...    k = 10,
    ... )

    >>> recommend.add(
    ...    documents=documents,
    ...    embeddings_documents=embeddings_documents,
    ...    embeddings_users=embeddings_users,
    ... )
    Recommend retriever
        key: id
        Users: 4
        Documents: 3

    >>> recommend += documents

    >>> print(recommend(user="Geoffrey"))
    [{'author': 'Montreal',
      'id': 'c',
      'similarity': 21229.834241369794,
      'title': 'Montreal'},
     {'author': 'Paris',
      'id': 'a',
      'similarity': 21229.634204933023,
      'title': 'Paris'},
     {'author': 'Madrid',
      'id': 'b',
      'similarity': 0.5075642957423536,
      'title': 'Madrid'}]


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
        index=None,
        store=MemoryStore(),
    ) -> None:
        super().__init__(key=key, on="", k=k)

        if index is None:
            self.index = Faiss(key=self.key)
        elif isinstance(index, Milvus) or isinstance(index, Faiss):
            self.index = index
        else:
            self.index = Faiss(key=self.key, index=index)

        self.store = store

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n \t key: {self.key}"
        repr += f"\n \t Users: {len(self.store)}"
        repr += f"\n \t Documents: {len(self.index)}"
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

        self.index.add(
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
        )

        self.store.add(
            users=[key for key in embeddings_users.keys()],
            embeddings=[embedding for embedding in embeddings_users.values()],
        )

        return self

    def __call__(
        self,
        user: typing.Union[str, int],
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
        """
        known, embedding, _ = self.store.get(values=[user])

        if not known:
            return []

        return self.index(
            **{
                "embedding": np.array(embedding),
                "k": self.k,
                "key": self.key,
                "expr": expr,
                "consistency_level": consistency_level,
                "partition_names": partition_names,
            }
        )
