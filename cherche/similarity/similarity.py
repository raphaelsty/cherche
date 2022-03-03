__all__ = ["cosine", "dot"]

import numpy as np


def cosine(emb_q: np.ndarray, emb_documents: list) -> list:
    """Computes cosine distance between input query embedding and documents embeddings.

    Bigger is better.

    Parameters
    ----------
    emb_q
        Embedding of the query.
    emb_documents
        List of embeddings of the documents.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import similarity

    >>> emb_q = np.array([1, 1])

    >>> emb_documents = [
    ...     np.array([0, 10]),
    ...     np.array([1, 1]),
    ... ]

    >>> print(similarity.cosine(emb_q=emb_q, emb_documents=emb_documents))
    [(1, 0.9999999999999998), (0, 0.7071067811865475)]

    """
    distances = {}
    for index, emb_document in enumerate(emb_documents):
        distances[index] = (emb_q @ emb_document) / (
            np.linalg.norm(emb_q) * np.linalg.norm(emb_document)
        )
    return [
        (index, float(distance))
        for index, distance in sorted(distances.items(), key=lambda item: item[1], reverse=True)
    ]


def dot(emb_q: np.ndarray, emb_documents: list) -> list:
    """Computes dot product between input query embedding and documents embeddings.

    Bigger is better.

    Parameters
    ----------
    emb_q
        Embedding of the query.
    emb_documents
        List of embeddings of the documents.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import similarity

    >>> emb_q = np.array([1, 1])

    >>> emb_documents = [
    ...     np.array([0, 10]),
    ...     np.array([1, 1]),
    ... ]

    >>> print(similarity.dot(emb_q=emb_q, emb_documents=emb_documents))
    [(0, 10), (1, 2)]

    """
    distances = {}
    for index, emb_document in enumerate(emb_documents):
        distances[index] = emb_q @ emb_document
    return [
        (index, float(distance))
        for index, distance in sorted(distances.items(), key=lambda item: item[1], reverse=True)
    ]
