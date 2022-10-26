import faiss
import numpy as np

__all__ = ["Faiss"]


class Faiss:
    """Faiss index dedicated to vector search.

    Parameters
    ----------
    key
        Identifier field for each document.
    index
        Faiss index to use.

    Examples
    --------
    >>> from cherche import index
    >>> from sentence_transformers import SentenceTransformer
    >>> from pprint import pprint as print

    >>> documents = [
    ...    {"id": 0, "title": "Paris"},
    ...    {"id": 1, "title": "Madrid"},
    ...    {"id": 2, "title": "Paris"}
    ... ]

    >>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    >>> faiss_index = index.Faiss(key="id")
    >>> faiss_index = faiss_index.add(
    ...    documents = documents,
    ...    embeddings = encoder.encode([document["title"] for document in documents]),
    ... )

    >>> print(faiss_index(embedding = encoder.encode(["Spain"])))
    [{'id': 1, 'similarity': 1.5076334135501044},
     {'id': 2, 'similarity': 0.9021741164485997},
     {'id': 0, 'similarity': 0.9021741164485997}]

    >>> documents = [
    ...    {"id": 3, "title": "Paris"},
    ...    {"id": 4, "title": "Madrid"},
    ...    {"id": 5, "title": "Paris"}
    ... ]

    >>> faiss_index = faiss_index.add(
    ...    documents = documents,
    ...    embeddings = encoder.encode([document["title"] for document in documents]),
    ... )

    >>> print(faiss_index(embedding = encoder.encode(["Spain"]), k=4))
    [{'id': 1, 'similarity': 1.5076334135501044},
     {'id': 4, 'similarity': 1.5076334135501044},
     {'id': 2, 'similarity': 0.9021741164485997},
     {'id': 3, 'similarity': 0.9021741164485997}]

    References
    ----------
    1. [Faiss](https://github.com/facebookresearch/faiss)

    """

    def __init__(self, key, index=None) -> None:
        self.key = key
        self.index = index
        self.ids: dict = {}
        self.documents: dict = {}

    def __len__(self) -> int:
        return len(self.ids)

    def _build(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build faiss index.

        Parameters
        ----------
        index
            faiss index.
        embeddings
            Embeddings of the documents.

        """
        array = np.array(embeddings).astype(np.float32)

        if self.index is None and embeddings:
            self.index = faiss.IndexFlatL2(array.shape[1])

        if not self.index.is_trained and embeddings:
            self.index.train(array)

        if embeddings:
            self.index.add(array)

        return self.index

    def add(self, documents: list, embeddings: np.ndarray) -> "Faiss":
        """Add documents to the faiss index and export embeddings if the path is provided.
        Streaming friendly.

        Parameters
        ----------
        documents
            List of documents as json or list of string to pre-compute queries embeddings.

        """
        n = 0
        index = len(self.documents)
        array = []

        for document, embedding in zip(documents, embeddings):
            # Skip the document if its id already exist.
            if document[self.key] in self.ids:
                continue

            self.documents[index + n] = {self.key: document[self.key]}
            self.ids[document[self.key]] = True
            array.append(embedding)
            n += 1

        self.index = self._build(embeddings=array)
        return self

    def __call__(self, embedding: np.ndarray, k: int = None, **kwargs) -> list:
        if k is None:
            k = len(self)

        distances, indexes = self.index.search(embedding, k)

        ranked = []
        for idx, distance in zip(indexes[0], distances[0]):

            if idx < 0:
                continue

            document = self.documents[idx]
            document["similarity"] = float(1 / distance) if distance > 0 else 0.0
            ranked.append(document)

        return ranked
