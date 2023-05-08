import typing

import numpy as np

from ..index import Faiss
from ..utils import yield_batch
from .base import Retriever

__all__ = ["Embedding"]


class Embedding(Retriever):
    """The Embedding retriever is dedicated to perform IR on embeddings calculated by the user
    rather than Cherche.

    Parameters
    ----------
    key
        Field identifier of each document.
    index
        Faiss index that will store the embeddings and perform the similarity search.
    normalize
        Whether to normalize the embeddings before adding them to the index in order to measure
        cosine similarity.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> recommend = retrieve.Embedding(
    ...    key="id",
    ... )

    >>> documents = [
    ...    {"id": "a", "title": "Paris", "author": "Paris"},
    ...    {"id": "b", "title": "Madrid", "author": "Madrid"},
    ...    {"id": "c", "title": "Montreal", "author": "Montreal"},
    ... ]

    >>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    >>> embeddings_documents = encoder.encode([
    ...    document["title"] for document in documents
    ... ])

    >>> recommend.add(
    ...    documents=documents,
    ...    embeddings_documents=embeddings_documents,
    ... )
    Embedding retriever
        key      : id
        documents: 3

    >>> queries = [
    ...    "Paris",
    ...    "Madrid",
    ...    "Montreal"
    ... ]

    >>> embeddings_queries = encoder.encode(queries)
    >>> print(recommend(embeddings_queries, k=2))
    [[{'id': 'a', 'similarity': 1.0},
      {'id': 'c', 'similarity': 0.5385907831761005}],
     [{'id': 'b', 'similarity': 1.0},
      {'id': 'a', 'similarity': 0.4990788711758875}],
     [{'id': 'c', 'similarity': 1.0},
      {'id': 'a', 'similarity': 0.5385907831761005}]]

    >>> embeddings_queries = encoder.encode("Paris")
    >>> print(recommend(embeddings_queries, k=2))
    [{'id': 'a', 'similarity': 0.9999999999989104},
     {'id': 'c', 'similarity': 0.5385907485958683}]

    """

    def __init__(
        self,
        key: str,
        index=None,
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 1024,
    ) -> None:
        super().__init__(key=key, on="", k=k, batch_size=batch_size)

        if index is None:
            self.index = Faiss(key=self.key, normalize=normalize)
        else:
            self.index = Faiss(key=self.key, index=index, normalize=normalize)

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n\tkey      : {self.key}"
        repr += f"\n\tdocuments: {len(self)}"
        return repr

    def __len__(self) -> int:
        return len(self.index)

    def add(
        self,
        documents: list,
        embeddings_documents: np.ndarray,
        **kwargs,
    ) -> "Embedding":
        """Add embeddings both documents and users.

        Parameters
        ----------
        documents
            List of documents to add to the index.

        embeddings_documents
            Embeddings of the documents ordered as the list of documents.
        """
        self.index.add(
            documents=documents,
            embeddings=embeddings_documents,
        )
        return self

    def __call__(
        self,
        q: np.ndarray,
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Retrieve documents from the index.

        Parameters
        ----------
        q
            Either a single query or a list of queries.
        k
            Number of documents to retrieve. Default is `None`, i.e all documents that match the
            query will be retrieved.
        batch_size
            Number of queries to encode at once.
        """
        k = k if k is not None else len(self)

        if len(q.shape) == 1:
            q = q.reshape(1, -1)

        rank = []
        for batch in yield_batch(
            array=q,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            desc=f"{self.__class__.__name__} retriever",
        ):
            rank.extend(
                self.index(
                    embeddings=batch,
                    k=k,
                )
            )

        return rank[0] if len(q) == 1 else rank
