import abc
import os
import pickle
import typing

import faiss
import numpy as np

from ..compose import Intersection, Pipeline, Union

__all__ = ["Retriever"]


class Retriever(abc.ABC):
    """Retriever base class.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.

    """

    def __init__(self, key: str, on: typing.Union[str, list], k: int) -> None:
        super().__init__()
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.k = k
        self.documents = None

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n \t key: {self.key}"
        repr += f"\n \t on: {', '.join(self.on)}"
        repr += f"\n \t documents: {self.__len__()}"
        return repr

    @property
    def type(self):
        return "retrieve"

    @abc.abstractclassmethod
    def __call__(self, q: str, **kwargs) -> list:
        return []

    def __len__(self):
        return len(self.documents) if self.documents is not None else 0

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, Pipeline):
            return Pipeline(self, other.models)
        elif isinstance(other, list):
            # Documents are part of the pipeline.
            return Pipeline([self, {document[self.key]: document for document in other}])
        return Pipeline([self, other])

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


class _BM25(Retriever):
    """Base class for BM25, BM25L and BM25Okapi Retriever.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Field to use to match the query to the documents.
    bm25
        BM25 model from [Rank BM25](https://github.com/dorianbrown/rank_bm25).
    tokenizer
        Tokenizer to use, the default one split on spaces. This tokenizer should have a
        `tokenizer.__call__` method that returns the list of tokenized tokens.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        documents: list,
        bm25,
        tokenizer=None,
        k: int = None,
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.bm25 = bm25
        self.tokenizer = tokenizer
        self.documents = {
            index: {self.key: document[self.key]} for index, document in enumerate(documents)
        }
        # Avoid adding documents to inversed index.
        self.ids = {}

    def __call__(self, q: str) -> list:
        """Retrieve the right document using BM25."""
        q = q.split(" ") if self.tokenizer is None else self.tokenizer(q)
        similarities = abs(self.model.get_scores(q))
        indexes, scores = [], []
        for index, score in enumerate(similarities):
            if score > 0:
                indexes.append(index)
                scores.append(score)

        # Empty
        if not indexes:
            return []

        scores, indexes = zip(*sorted(zip(scores, indexes), reverse=True))
        documents = [
            {**self.documents[index], "similarity": float(score)}
            for index, score in zip(indexes, scores)
        ]
        return documents[: self.k] if self.k is not None else documents

    def _process_documents(self, documents: list) -> list:
        """Documents to feed BM25 retriever."""
        bm25_documents = []
        for doc in documents:
            doc = " ".join([doc.get(field, "") for field in self.on])
            if self.tokenizer is None:
                doc = doc.split(" ")
            else:
                doc = self.tokenizer(doc)
            bm25_documents.append(doc)
        return bm25_documents


class BaseEncoder(Retriever):
    def __init__(
        self, encoder, key: str, on: typing.Union[str, list], k: int, path: str, query_encoder=None
    ) -> None:
        super().__init__(key=key, on=on, k=k)
        self.encoder = encoder
        self.path = path
        self.tree = None
        self.documents = {}
        self.q_embeddings = {}
        self.query_encoder = query_encoder
        self.ids = {}

    @staticmethod
    def build_faiss(tree: faiss.IndexFlatL2, documents_embeddings: list) -> faiss.IndexFlatL2:
        """Build faiss index.

        Parameters
        ----------
        tree
            faiss index.
        documents_embeddings
            Embeddings of the documents.

        """
        array_embeddings = np.array(documents_embeddings).astype(np.float32)
        if tree is None and documents_embeddings:
            tree = faiss.IndexFlatL2(array_embeddings.shape[1])
        # Check that documents embeddings is not empty
        if documents_embeddings:
            tree.add(array_embeddings)
        return tree

    @staticmethod
    def load_embeddings(path: str) -> dict:
        """Load embeddings from an existing directory."""
        if path is None:
            return {}
        if not os.path.isfile(path):
            return {}
        with open(path, "rb") as input_embeddings:
            embeddings = pickle.load(input_embeddings)
        return embeddings

    @staticmethod
    def dump_embeddings(embeddings: dict, path: str) -> None:
        """Dump embeddings to the selected directory."""
        if path is not None:
            with open(path, "wb") as ouput_embeddings:
                pickle.dump(embeddings, ouput_embeddings)

    def add(self, documents: list) -> "BaseEncoder":
        """Add documents to the faiss index and export embeddings if the path is provided.
        Streaming friendly.

        Parameters
        ----------
        documents
            List of documents as json or list of string to pre-compute queries embeddings.

        """
        embeddings = self.load_embeddings(path=self.path)

        # Pre-compute query embeddings
        if isinstance(documents[0], str):

            query_encoder = self.query_encoder if self.query_encoder is not None else self.encoder

            for query, embedding in zip(
                documents,
                query_encoder(
                    [
                        document
                        for document in documents
                        if isinstance(document, str) and document not in self.q_embeddings
                    ]
                ),
            ):
                embeddings[query] = embedding
                self.q_embeddings[query] = embedding

                if self.path is not None:
                    self.dump_embeddings(embeddings=embeddings, path=self.path)
                    return self

        # Pre-compute documents embeddings
        # Create index between faiss and documents ids.
        n = len(self.documents)
        self.documents.update(
            {
                index + n: {self.key: document[self.key]}
                for index, document in enumerate(documents)
                if document[self.key] not in self.ids
            }
        )

        # Pre-compute documents embeddings and index them using Faiss
        keys, new_documents = [], []
        for document in documents:
            if document[self.key] not in embeddings and str(document[self.key]) not in embeddings:
                keys.append(document[self.key])
                new_documents.append(" ".join([document.get(field, "") for field in self.on]))

        for key, embedding in zip(keys, self.encoder(new_documents)):
            embeddings[key] = embedding

        if self.path is not None:
            self.dump_embeddings(embeddings=embeddings, path=self.path)

        self.tree = self.build_faiss(
            tree=self.tree,
            documents_embeddings=[
                embeddings[document[self.key]]
                for document in documents
                if document[self.key] not in self.ids
            ],
        )

        # Avoid duplicate documents.
        self.ids.update({document[self.key]: True for document in documents})
        return self
