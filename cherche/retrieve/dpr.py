__all__ = ["DPR"]

import typing

import more_itertools
import tqdm

from ..index import Faiss, Milvus
from .base import Retriever


class DPR(Retriever):
    """DPR as a retriever using Faiss Index.

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

    >>> retriever = retrieve.DPR(
    ...    key = "id",
    ...    on = ["title", "author"],
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
    ...    k = 2,
    ... )

    >>> retriever.add(documents)
    DPR retriever
         key: id
         on: title, author
         documents: 3

    >>> print(retriever("Spain"))
    [{'id': 1, 'similarity': 0.009192565994771673},
     {'id': 0, 'similarity': 0.008331424302852155}]

    >>> retriever += documents

    >>> print(retriever("Spain"))
    [{'author': 'Madrid',
      'id': 1,
      'similarity': 0.009192565994771673,
      'title': 'Madrid'},
     {'author': 'Paris',
      'id': 0,
      'similarity': 0.008331424302852155,
      'title': 'Paris'}]

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        k: int,
        encoder,
        query_encoder,
        index=None,
        path: str = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.encoder = encoder
        self.query_encoder = query_encoder

        if index is None:
            self.index = Faiss(key=self.key)
        elif isinstance(index, Milvus) or isinstance(index, Faiss):
            self.index = index
        else:
            self.index = Faiss(key=self.key, index=index)

    def __len__(self) -> int:
        return len(self.index)

    def add(self, documents: list, batch_size: int = 64, **kwargs) -> "DPR":
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
            desc="Retriever embeddings calculation.",
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

    def __call__(
        self,
        q: str,
        expr: str = None,
        consistency_level: str = None,
        partition_names: list = None,
        **kwargs
    ) -> list:
        """Search for documents.

        Parameters
        ----------
        q
            Query.
        """
        return self.index(
            **{
                "embedding": self.query_encoder([q]),
                "k": self.k,
                "key": self.key,
                "expr": expr,
                "consistency_level": consistency_level,
                "partition_names": partition_names,
            }
        )
