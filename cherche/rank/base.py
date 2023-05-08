__all__ = ["Ranker"]

import abc
import collections
import typing

import numpy as np
import tqdm

from ..compose import Intersection, Pipeline, Union, Vote
from ..utils import yield_batch


class MemoryStore:
    """Store embeddings of rankers in memory.

    Parameters
    ----------
    key
        Key to use to store the embeddings in memory.

    """

    def __init__(self, key: str) -> None:
        self.key = key
        self.embeddings = {}

    def __len__(self) -> int:
        return len(self.embeddings)

    def add(
        self,
        embeddings: typing.List[np.ndarray],
        documents: typing.List[typing.Dict[str, str]],
        **kwargs,
    ) -> "MemoryStore":
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
        documents
            List of documents or list of string for embeddings pre-comptuting.

        """
        for document, embedding in zip(documents, embeddings):
            self.embeddings[document[self.key]] = embedding
        return self

    def get(
        self,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        **kwargs,
    ) -> typing.Tuple[
        typing.List[str],
        typing.List[np.ndarray],
        typing.List[typing.Dict[str, str]],
    ]:
        """Distinguish known documents with their embeddings from unknown documents."""
        known, embeddings, unknown = [], [], []
        for batch in documents:
            for document in batch:
                key = document[self.key]
                if key in self.embeddings:
                    known.append(key)
                    embeddings.append(self.embeddings[key])
                else:
                    unknown.append(document)
        return known, embeddings, unknown


class Ranker(abc.ABC):
    """Abstract class for ranking models.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields of the documents to use for ranking.
    encoder
        Encoding function to computes embeddings of the documents.
    normalize
        Normalize the embeddings in order to measure cosine similarity if set to True, dot product
        if set to False.
    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, typing.List[str]],
        encoder,
        normalize: bool,
        batch_size: int,
        k: typing.Optional[int] = None,
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.encoder = encoder
        self.store = MemoryStore(key=self.key)
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size

    def __len__(self):
        return len(self.store)

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n\tkey       : {self.key}"
        repr += f"\n\ton        : {', '.join(self.on)}"
        repr += f"\n\tnormalize : {self.normalize}"
        repr += f"\n\tembeddings: {len(self.store)}"
        return repr

    @abc.abstractmethod
    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: int,
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Rank documents according to the query."""
        if isinstance(q, str):
            return []
        elif isinstance(q, list):
            return [[]]

    def _encoder(self, documents: typing.List[typing.Dict[str, str]]) -> np.ndarray:
        """Computes documents embeddings."""
        return self.encoder(
            [
                " ".join([document.get(field, "") for field in self.on])
                for document in documents
            ]
        )

    def _batch_encode(
        self, documents: typing.List[typing.Dict[str, str]], batch_size: int, desc: str
    ) -> typing.List[np.ndarray]:
        """Computes documents embeddings per batch."""
        embeddings = []
        for batch in yield_batch(
            array=documents,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} ranker",
        ):
            embeddings.extend(self._encoder(documents=batch))
        return embeddings

    def add(
        self, documents: typing.List[typing.Dict[str, str]], batch_size: int = 64
    ) -> "Ranker":
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
        documents
            List of documents or list of string for embeddings pre-comptuting.

        """
        self.store.add(
            documents=documents,
            embeddings=self._batch_encode(
                documents=documents,
                batch_size=batch_size,
                desc=f"{self.__class__.__name__} index creation",
            ),
        )
        return self

    def _encode(
        self,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        batch_size: typing.Optional[int] = None,
    ) -> typing.Dict[str, np.ndarray]:
        """Computes documents embeddings if they are not in the store."""
        known, embeddings, unknown = self.store.get(documents=documents)
        if unknown:
            # Encode unknown documents
            unknown_embeddings = self._batch_encode(
                documents=unknown,
                batch_size=batch_size,
                desc=f"{self.__class__.__name__} missing index documents",
            )

            # Merge known and unknown documents
            known += [document[self.key] for document in unknown]
            embeddings = embeddings + unknown_embeddings

        return {key: embedding for key, embedding in zip(known, embeddings)}

    def rank(
        self,
        embeddings_documents: typing.Dict[str, np.ndarray],
        embeddings_queries: np.ndarray,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        k: int,
        batch_size: typing.Optional[int] = None,
    ) -> list:
        """Rank inputs documents ordered by relevance among the top k.

        Parameters
        ----------
        embeddings_queries
            Embedding of the queries.
        embeddings_documents
            Embeddings of the documents.
        documents
            List of documents to re-rank.
        k
            Number of documents to keep.
        batch_size
            Batch size for encoding documents.

        """
        # Reshape query embeddings if needed
        if len(embeddings_queries.shape) == 1:
            embeddings_queries = embeddings_queries.reshape(1, -1)

        # Normalize embeddings to compute cosine similarity
        if self.normalize:
            embeddings_queries = (
                embeddings_queries
                / np.linalg.norm(embeddings_queries, axis=-1)[:, None]
            )

        # Compute scores.
        scores, missing = [], []
        for q, batch in tqdm.tqdm(
            zip(embeddings_queries, documents), position=0, desc="Ranker scoring"
        ):
            if batch:
                scores.append(
                    q
                    @ np.stack(
                        [embeddings_documents[d[self.key]] for d in batch], axis=0
                    ).T
                )
                missing.append(False)
            else:
                # Retriever did not found any document for the query
                scores.append(np.array([]))
                missing.append(True)

        ranked = []
        for scores_query, documents_query, missing_query in tqdm.tqdm(
            zip(scores, documents, missing), position=0, desc="Ranker sorting"
        ):
            if missing_query:
                ranked.append([])
                continue

            scores_query = scores_query.reshape(1, -1)
            ranks_query = np.fliplr(np.argsort(scores_query))
            scores_query, ranks_query = scores_query.flatten(), ranks_query.flatten()
            ranks_query = ranks_query[:k]
            ranked.append(
                [
                    {
                        **document,
                        "similarity": similarity,
                    }
                    for document, similarity in zip(
                        np.take(documents_query, ranks_query),
                        np.take(scores_query, ranks_query),
                    )
                ]
            )

        return ranked

    def encode_rank(
        self,
        embeddings_queries: np.ndarray,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        k: int,
        batch_size: typing.Optional[int] = None,
    ) -> typing.List[typing.List[typing.Dict[str, str]]]:
        """Encode documents and rank them according to the query."""
        embeddings_documents = self._encode(documents=documents, batch_size=batch_size)
        return self.rank(
            embeddings_documents=embeddings_documents,
            embeddings_queries=embeddings_queries,
            documents=documents,
            k=k,
            batch_size=batch_size,
        )

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, Pipeline):
            return other + self
        elif isinstance(other, list):
            # Documents are part of the pipeline.
            return Pipeline(
                [self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[other, self])

    def __or__(self, other) -> Union:
        """Union operator."""
        if isinstance(other, Union):
            return Union([self] + other.models)
        return Union([self, other])

    def __and__(self, other) -> Intersection:
        """Intersection operator."""
        if isinstance(other, Intersection):
            return Intersection([self] + other.models)
        return Intersection([self, other])

    def __mul__(self, other) -> Vote:
        """Voting operator."""
        if isinstance(other, Vote):
            return Vote([self] + other.models)
        return Vote([self, other])
