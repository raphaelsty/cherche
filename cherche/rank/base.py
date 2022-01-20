__all__ = ["Ranker"]

import abc
import os
import pickle
import typing

from ..compose import Intersection, Pipeline, Union


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
        self, key: str, on: typing.Union[str, list], encoder, k: int, path: str, similarity
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.encoder = encoder
        self.k = k
        self.path = path
        self.similarity = similarity
        self.embeddings = self.load_embeddings(path=path) if self.path is not None else {}

    @property
    def type(self):
        return "rank"

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n\t key: {self.key}"
        repr += f"\n\t on: {', '.join(self.on)}"
        repr += f"\n\t k: {self.k}"
        repr += f"\n\t similarity: {self.similarity.__name__}"
        if self.path is not None:
            repr += f"\n\t embeddings stored at: {self.path}"

        return repr

    @abc.abstractmethod
    def __call__(self, q: str, documents: list, **kwargs) -> list:
        if not documents:
            return []
        return self

    def add(self, documents: list) -> "Ranker":
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
        documents
            List of documents or list of string for embeddings pre-comptuting.

        """
        documents_rankers = []
        keys = []
        for document in documents:
            if isinstance(document, str):
                self.embeddings[document] = self.encoder(document)

            elif document[self.key] not in self.embeddings:
                keys.append(document[self.key])
                documents_rankers.append(" ".join([document.get(field, "") for field in self.on]))

        if documents_rankers:
            for key, embedding in zip(keys, self.encoder(documents_rankers)):
                self.embeddings[key] = embedding

            if self.path is not None:
                self.dump_embeddings(embeddings=self.embeddings, path=self.path)

        return self

    def _emb_documents(self, documents: list) -> list:
        """Computes documents embeddings."""
        emb_documents = []
        for document in documents:
            # ElasticSearch can store embeddings
            if "embedding" in document:
                embedding = document.pop("embedding")
            else:
                if document[self.key] in self.embeddings:
                    embedding = self.embeddings[document[self.key]]
                else:
                    embedding = self.encoder(
                        " ".join([document.get(field, "") for field in self.on])
                    )
            emb_documents.append(embedding)
        return emb_documents

    def _rank(self, similarities: list, documents: list) -> list:
        """Rank inputs documents ordered by relevance among the top k.

        Parameters
        ----------
        similarities
            List of tuples (index, similarity) among the list of documents to rank.
        documents
            List of documents.

        """
        similarities = similarities[: self.k] if self.k is not None else similarities
        ranked = []
        for index, similarity in similarities:
            document = documents[index]
            document["similarity"] = similarity
            ranked.append(document)
        return ranked

    @staticmethod
    def load_embeddings(path: str) -> dict:
        """Load embeddings from an existing directory."""
        if not os.path.isfile(path):
            return {}
        with open(path, "rb") as input_embeddings:
            embeddings = pickle.load(input_embeddings)
        return embeddings

    @staticmethod
    def dump_embeddings(embeddings, path: str) -> None:
        """Dump embeddings to the selected directory."""
        with open(path, "wb") as ouput_embeddings:
            pickle.dump(embeddings, ouput_embeddings)

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, Pipeline):
            return other + self
        elif isinstance(other, list):
            # Documents are part of the pipeline.
            return Pipeline([self, {document[self.key]: document for document in other}])
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

    def embs(self, documents: list) -> list:
        """Computes and returns embeddings of input documents.

        Parameters
        ----------
        documents
            List of documents for whiwh to computes embeddings.

        """
        return self.encoder(
            [" ".join([doc.get(field, "") for field in self.on]) for doc in documents]
        )
