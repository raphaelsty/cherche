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
        self, on: typing.Union[str, list], encoder, k: int, path: str, similarity
    ) -> None:
        self.on = on if isinstance(on, list) else [on]
        self.encoder = encoder
        self.k = k
        self.path = path
        self.similarity = similarity
        self.embeddings = self.load_embeddings(path=path) if self.path is not None else {}

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
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
        for document in documents:
            if isinstance(document, dict):
                document = " ".join([document[field] for field in self.on])
            if document not in self.embeddings:
                documents_rankers.append(document)

        if documents_rankers:
            for document, embedding in zip(documents_rankers, self.encoder(documents_rankers)):
                self.embeddings[document] = embedding

            if self.path is not None:
                self.dump_embeddings(embeddings=self.embeddings, path=self.path)

        return self

    def _emb_documents(self, documents: list) -> list:
        """Computes documents embeddings."""
        emb_documents = []
        for document in documents:
            document = " ".join([document[field] for field in self.on])
            # ElasticSearch can store embeddings
            if "embedding" in document:
                embedding = document.pop("embedding")
            elif document in self.embeddings:
                embedding = self.embeddings[document]
            else:
                embedding = self.encoder(document)
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
        else:
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
        return self.encoder([" ".join([doc[field] for field in self.on]) for doc in documents])
