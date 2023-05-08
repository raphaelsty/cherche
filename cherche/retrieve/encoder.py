__all__ = ["Encoder"]

import typing

import tqdm

from ..index import Faiss
from ..utils import yield_batch
from .base import Retriever


class Encoder(Retriever):
    """Encoder as a retriever using Faiss Index.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Field to use to retrieve documents.
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

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> retriever = retrieve.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title"],
    ... )

    >>> retriever.add(documents, batch_size=1)
    Encoder retriever
        key      : id
        on       : title
        documents: 3

    >>> print(retriever("Spain", k=2))
    [{'id': 1, 'similarity': 0.6544566453117681},
     {'id': 0, 'similarity': 0.5405465419981407}]

    >>> print(retriever(["Spain", "Montreal"], k=2))
    [[{'id': 1, 'similarity': 0.6544566453117681},
      {'id': 0, 'similarity': 0.54054659424589}],
     [{'id': 2, 'similarity': 0.7372165680578416},
      {'id': 0, 'similarity': 0.5185645704259234}]]

    """

    def __init__(
        self,
        encoder,
        key: str,
        on: typing.Union[str, list],
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 64,
        index=None,
    ) -> None:
        super().__init__(
            key=key,
            on=on,
            k=k,
            batch_size=batch_size,
        )
        self.encoder = encoder

        if index is None:
            self.index = Faiss(key=self.key, normalize=normalize)
        else:
            self.index = Faiss(key=self.key, index=index, normalize=normalize)

    def __len__(self) -> int:
        return len(self.index)

    def add(
        self,
        documents: typing.List[typing.Dict[str, str]],
        batch_size: int = 64,
        **kwargs,
    ) -> "Encoder":
        """Add documents to the index.

        Parameters
        ----------
        documents
            List of documents to add to the index.
        batch_size
            Number of documents to encode at once.
        """

        for batch in yield_batch(
            array=documents,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} index creation",
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
        q: typing.Union[typing.List[str], str],
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

        rank = []
        for batch in yield_batch(
            array=q,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            desc=f"{self.__class__.__name__} retriever",
        ):
            rank.extend(
                self.index(
                    embeddings=self.encoder(batch),
                    k=k,
                )
            )

        return rank[0] if isinstance(q, str) else rank
