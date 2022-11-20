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
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-multiset-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-multiset-base').encode,
    ...    k = 2,
    ... )

    >>> retriever.add(documents)
    DPR retriever
         key: id
         on: title, author
         documents: 3

    >>> print(retriever("Spain"))
    [{'id': 1, 'similarity': 0.006858040774451286},
     {'id': 0, 'similarity': 0.0060191201849380555}]

    >>> print(retriever("Paris"))
    [{'id': 0, 'similarity': 0.00816787668669813},
     {'id': 1, 'similarity': 0.007023785549903056}]

    >>> print(retriever.batch(["Spain", "Paris"]))
    {0: [{'id': 1, 'similarity': 0.006858040774451286},
         {'id': 0, 'similarity': 0.006019121290584248}],
     1: [{'id': 0, 'similarity': 0.008167878213665493},
         {'id': 1, 'similarity': 0.007023786302673574}]}

    >>> print(retriever.batch(["Spain", "Paris"], batch_size=1))
    {0: [{'id': 1, 'similarity': 0.006858040774451286},
         {'id': 0, 'similarity': 0.0060191201849380555}],
     1: [{'id': 0, 'similarity': 0.00816787668669813},
         {'id': 1, 'similarity': 0.007023785549903056}]}

    >>> retriever += documents

    >>> print(retriever("Spain"))
    [{'author': 'Madrid',
      'id': 1,
      'similarity': 0.006858040774451286,
      'title': 'Madrid'},
     {'author': 'Paris',
      'id': 0,
      'similarity': 0.0060191201849380555,
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

    def batch(
        self,
        q: typing.List[str],
        batch_size: int = 64,
        expr: str = None,
        consistency_level: str = None,
        partition_names: list = None,
        **kwargs
    ) -> dict:
        """Search for documents per batch.

        Parameters
        ----------
        q
            List of queries.
        """
        rank = {}

        for batch in tqdm.tqdm(
            more_itertools.chunked(q, batch_size),
            position=0,
            desc="Retriever batch queries.",
            total=1 + len(q) // batch_size,
        ):

            rank = {
                **rank,
                **self.index.batch(
                    **{
                        "embeddings": self.query_encoder(batch),
                        "k": self.k,
                        "n": len(rank),
                        "key": self.key,
                        "expr": expr,
                        "consistency_level": consistency_level,
                        "partition_names": partition_names,
                    }
                ),
            }

        return rank
