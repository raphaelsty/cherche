__all__ = ["Ranker"]

import abc
import os
import pickle

from ..metric import cosine_distance


class Ranker(abc.ABC):
    """Abstract class for ranking models.

    Parameters
    ----------

        on: Field of the documents to use for ranking.
        encoder: Encoding function to computes embeddings of the documents.
        path: Path of the file dedicated to store the embeddings as a pickle file.
        metric: Distance / similarity measure to use i.e cherche.metric.cosine_distance or
            cherche.metric.dot_similarity.

    """

    def __init__(self, on: str, encoder, path: str = None, metric=cosine_distance) -> None:
        self.on = on
        self.encoder = encoder
        self.path = path
        self.metric = metric
        self.embeddings = self.load_embeddings(path=path) if self.path is not None else {}

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n \t on: {self.on}"
        return repr

    @abc.abstractmethod
    def __call__(self, q: str, documents: list, k: int = None) -> list:
        pass

    def add(self, documents):
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
            documents: List of documents.

        """
        documents = [
            document[self.on] for document in documents if document[self.on] not in self.embeddings
        ]

        for document, embedding in zip(documents, self.encoder(documents)):
            self.embeddings[document] = embedding

        if self.path is not None:
            self.dump_embeddings(embeddings=self.embeddings, path=self.path)

        return self

    def _rank(self, distances: list, documents: list, k: int = None):
        """Rank inputs documents ordered by relevance among the top k.

        Parameters
        ----------

            distances: List of tuples of index among the list of documents and distances.
            documents: List of documents.
            k: Number of documents to keep.

        """
        distances = distances[:k] if k is not None else k
        ranked = []
        for index, distance in distances:
            document = documents[index]
            document[self.metric.__name__] = distance
            ranked.append(document)
        return ranked

    @staticmethod
    def load_embeddings(path: str):
        """Load embeddings from an existing directory."""
        if not os.path.isfile(path):
            return {}
        with open(path, "rb") as input_embeddings:
            embeddings = pickle.load(input_embeddings)
        return embeddings

    @staticmethod
    def dump_embeddings(embeddings, path: str):
        """Dump embeddings to the selected directory."""
        with open(path, "wb") as ouput_embeddings:
            pickle.dump(embeddings, ouput_embeddings)
