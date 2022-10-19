__all__ = ["Encoder"]

import typing

import faiss
import more_itertools
import tqdm

from ..index import Faiss
from .base import Retriever


class Encoder(Retriever):
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
    index
        Index that will store the embeddings and perform the similarity search. The default
        index is Faiss.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "author": "Paris"},
    ...    {"id": 1, "title": "Madrid", "author": "Madrid"},
    ...    {"id": 2, "title": "Montreal", "author": "Montreal"},
    ... ]

    >>> retriever = retrieve.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title", "author"],
    ...    k = 2,
    ... )

    >>> retriever.add(documents)
    Encoder retriever
         key: id
         on: title, author
         documents: 3

    >>> print(retriever("Spain"))
    [{'id': 1, 'similarity': 1.1885032405192992},
     {'id': 0, 'similarity': 0.8492543139964137}]

    References
    ----------
    1. [Faiss](https://github.com/facebookresearch/faiss)

    """

    def __init__(
        self,
        encoder,
        key: str,
        on: typing.Union[str, list],
        k: int,
        index=None,
        path: str = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.encoder = encoder

        if index is None:
            self.index = Faiss(key=self.key)
        elif isinstance(index, faiss):
            self.index = Faiss(key=self.key, index=index)

    def __len__(self) -> int:
        return len(self.index)

    def add(self, documents: list, batch_size: int = 64, **kwargs) -> "Encoder":
        """Add documents to the index.

        Parameters
        ----------
        documents
            List of documents.
        batch_size
            Batch size to be encoded.
        """
        for batch in tqdm.tqdm(
            more_itertools.chunked(documents, batch_size),
            position=0,
            desc="Embeddings calculation.",
            total=1 + len(documents) // batch_size,
        ):
            self.index.add(
                documents=batch,
                embeddings=self.encoder(
                    [
                        " ".join([document.get(field, "") for field in self.on])
                        for document in batch
                    ]
                ),
            )
        return self

    def __call__(self, q: str) -> list:
        """Search for documents.

        Parameters
        ----------
        q
            Query.
        """
        return self.index(embedding=self.encoder([q]), k=self.k)
