__all__ = ["Encoder"]

import typing

import numpy as np

from .base import BaseEncoder


class Encoder(BaseEncoder):
    """Encoder as a retriever using Faiss Index.

    Parameters
    ----------
    on
        Field to use to retrieve documents.
    k
        Number of documents to retrieve.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    on = ["title", "article"],
    ...    k = 2,
    ...    path = "retriever_encoder.pkl"
    ... )

    >>> retriever.add(documents)
    Encoder retriever
         on: title, article
         documents: 3

    >>> print(retriever("Paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'similarity': 1.472814254853544,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'similarity': 1.0293491728070765,
      'title': 'Eiffel tower'}]

    >>> retriever.add(documents)
    Encoder retriever
         on: title, article
         documents: 6

    >>> print(retriever("Paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'similarity': 1.472814254853544,
      'title': 'Paris'},
     {'article': 'This town is the capital of France',
      'author': 'Wiki',
      'similarity': 1.472814254853544,
      'title': 'Paris'}]

    References
    ----------
    1. [Faiss](https://github.com/facebookresearch/faiss)

    """

    def __init__(self, encoder, on: typing.Union[str, list], k: int, path: str = None) -> None:
        super().__init__(encoder=encoder, on=on, k=k, path=path)
        self.embeddings = self.load_embeddings(path=self.path)

    def __call__(self, q: str) -> list:
        distances, indexes = self.tree.search(
            np.array([self.encoder(q) if q not in self.embeddings else self.embeddings[q]]).astype(
                np.float32
            ),
            self.k if self.k is not None else len(self.documents),
        )
        ranked = []
        for index, distance in zip(indexes[0], distances[0]):
            document = self.documents[index]
            document["similarity"] = 1 / distance
            ranked.append(document)
        return ranked

    def add(self, documents: list) -> "Encoder":
        """Add documents to the faiss index and export embeddings if the path is provided.

        Parameters
        ----------
        documents
            List of documents as json or list of string to pre-compute queries embeddings.

        """
        self.documents += documents

        # Pre-compute query embeddings
        if isinstance(documents[0], str):
            for query, embedding in zip(
                documents,
                self.encoder(
                    [
                        document
                        for document in documents
                        if isinstance(document, str) and document not in self.embeddings
                    ]
                ),
            ):
                self.embeddings[query] = embedding

        # Pre-compute documents embeddings and index them using Faiss
        new_documents = []
        for document in documents:
            document = " ".join([document[field] for field in self.on])
            if document not in self.embeddings:
                new_documents.append(document)

        for document, embedding in zip(new_documents, self.encoder(new_documents)):
            self.embeddings[document] = embedding

        if self.path is not None:
            self.dump_embeddings(embeddings=self.embeddings, path=self.path)

        self.tree = self.build_faiss(
            tree=self.tree,
            documents_embeddings=[
                self.embeddings[" ".join([document[field] for field in self.on])]
                for document in documents
            ],
        )
        return self
