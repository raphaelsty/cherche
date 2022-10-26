from __future__ import annotations

__all__ = ["Ranker"]

import abc
import typing

import more_itertools
import numpy as np
import tqdm

from ..compose import Intersection, Pipeline, Union, Vote


class MemoryStore:
    """Store embeddings of rankers in memory."""

    def __init__(self) -> None:
        self.embeddings = {}

    def __len__(self) -> int:
        return len(self.embeddings)

    def add(
        self,
        embeddings: list,
        key: str = None,
        documents: list = None,
        users=None,
        **kwargs,
    ) -> "MemoryStore":
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
        documents
            List of documents or list of string for embeddings pre-comptuting.

        """
        if users is not None:
            for user, embedding in zip(users, embeddings):
                self.embeddings[user] = np.array(embedding).flatten()
        elif documents is not None:
            for document, embedding in zip(documents, embeddings):
                self.embeddings[document[key]] = np.array(embedding).flatten()
        return self

    def get(self, values: list, **kwargs) -> typing.Tuple[list, list, list]:
        """Get specific embeddings from documents ids."""
        known, embeddings, unknown = [], [], []
        for key in values:
            embedding = self.embeddings.get(key, None)
            if embedding is not None:
                known.append(key)
                embeddings.append(embedding)
            else:
                unknown.append(key)
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
    k
        Number of documents to keep.
    path
        Path of the file dedicated to store the embeddings as a pickle file.
    similarity
        Similarity measure to use i.e similarity.cosine or similarity.dot.

    """

    def __init__(
        self,
        key: str,
        on: str | list,
        encoder,
        k: int,
        similarity,
        store,
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.k = k
        self.encoder = encoder
        self.similarity = similarity
        self.store = store

    @property
    def type(self):
        return "rank"

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n\t key: {self.key}"
        repr += f"\n\t on: {', '.join(self.on)}"
        repr += f"\n\t k: {self.k}"
        repr += f"\n\t similarity: {self.similarity.__name__}"
        repr += f"\n\t Embeddings pre-computed: {len(self.store)}"
        return repr

    @abc.abstractmethod
    def __call__(self, q: str, documents: list, **kwargs) -> list:
        return []

    def add(self, documents: list, batch_size: int = 64) -> "Ranker":
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
        documents
            List of documents or list of string for embeddings pre-comptuting.

        """
        for batch in tqdm.tqdm(
            more_itertools.chunked(documents, batch_size),
            position=0,
            desc="Ranker embeddings calculation.",
            total=1 + len(documents) // batch_size,
        ):
            self.store.add(
                **{
                    "key": self.key,
                    "documents": batch,
                    "embeddings": self.encoder(
                        [
                            " ".join([document.get(field, "") for field in self.on])
                            for document in batch
                        ]
                    ),
                }
            )
        return self

    def encode(self, documents: list) -> typing.Tuple[dict, list]:
        """Computes documents embeddings."""
        known, embeddings, unknown = self.store.get(
            **{
                "key": self.key,
                "values": [document[self.key] for document in documents],
            }
        )
        index = {document[self.key]: document for document in documents}

        embeddings_unknown = self.encoder(
            [
                " ".join([index[key_unknown].get(field, "") for field in self.on])
                for key_unknown in unknown
            ]
        )

        return {
            idx: index.get(key_document)
            for idx, key_document in enumerate(known + unknown)
        }, np.array(list(embeddings) + list(embeddings_unknown))

    def rank(self, query_embedding: np.ndarray, documents: list) -> list:
        """Rank inputs documents ordered by relevance among the top k.

        Parameters
        ----------
        similarities
            List of tuples (index, similarity) among the list of documents to rank.
        documents
            List of documents.

        """
        index, embeddings = self.encode(documents=documents)
        similarities = self.similarity(emb_q=query_embedding, emb_documents=embeddings)
        similarities = similarities[: self.k] if self.k is not None else similarities
        ranked = []
        for idx, similarity in similarities:
            document = index[idx]
            document["similarity"] = similarity
            ranked.append(document)
        return ranked

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
