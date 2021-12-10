__all__ = ["Ranker"]

import abc
import os
import pickle

from ..compose import Compose


class Ranker(abc.ABC):
    """Abstract class for ranking models.

    Parameters
    ----------

        on: Field of the documents to use for ranking.
        encoder: Encoding function to computes embeddings of the documents.
        k: Number of documents to keep.
        path: Path of the file dedicated to store the embeddings as a pickle file.
        distance: Distance / similarity measure to use i.e cherche.distance.cosine_distance or
            cherche.distance.dot_similarity.

    """

    def __init__(self, on: str, encoder, k: int, path: str, distance) -> None:
        self.on = on
        self.encoder = encoder
        self.k = k
        self.path = path
        self.distance = distance
        self.embeddings = self.load_embeddings(path=path) if self.path is not None else {}

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n\t on: {self.on}"
        repr += f"\n\t k: {self.k}"
        repr += f"\n\t distance: {self.distance.__name__}"
        if self.path is not None:
            repr += f"\n\t embeddings stored at: {self.path}"

        return repr

    @abc.abstractmethod
    def __call__(self, q: str, documents: list, **kwargs) -> list:
        if not documents:
            return []
        return self

    def add(self, documents):
        """Pre-compute embeddings and store them at the selected path.

        Parameters
        ----------
            documents: List of documents.

        """
        documents = [
            document[self.on] for document in documents if document[self.on] not in self.embeddings
        ]

        if documents:
            for document, embedding in zip(documents, self.encoder(documents)):
                self.embeddings[document] = embedding

            if self.path is not None:
                self.dump_embeddings(embeddings=self.embeddings, path=self.path)

        return self

    def _rank(self, distances: list, documents: list):
        """Rank inputs documents ordered by relevance among the top k.

        Parameters
        ----------

            distances: List of tuples (index, distances) among the list of documents to rank.
            documents: List of documents.

        """
        distances = distances[: self.k] if self.k is not None else distances
        ranked = []
        for index, distance in distances:
            document = documents[index]
            document[self.distance.__name__] = distance
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

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Compose):
            return other + self
        else:
            return Compose(models=[other, self])
