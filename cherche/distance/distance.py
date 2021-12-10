__all__ = ["cosine_distance", "dot_similarity"]

import numpy as np
from scipy.spatial import distance


def cosine_distance(emb_q: np.ndarray, emb_documents: list):
    """Computes cosine distance between input query embedding and documents embeddings.
    Lower is better.

    Parameters
    ----------

        emb_q: Embedding of the query.
        emb_documents: List of embeddings of the documents.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import distance

    >>> emb_q = np.array([1, 1])

    >>> emb_documents = [
    ...     np.array([0, 10]),
    ...     np.array([1, 1]),
    ... ]

    >>> print(distance.cosine_distance(emb_q=emb_q, emb_documents=emb_documents))
    [(1, 0.0), (0, 0.29289321881345254)]

    """
    distances = {}
    for index, emb_document in enumerate(emb_documents):
        distances[index] = distance.cosine(emb_q, emb_document)
    return [
        (index, distance)
        for index, distance in sorted(distances.items(), key=lambda item: item[1])
    ]


def dot_similarity(emb_q: np.ndarray, emb_documents: list):
    """Computes dot product between input query embedding and documents embeddings.
    Higher is better.

    Parameters
    ----------

        emb_q: Embedding of the query.
        emb_documents: List of embeddings of the documents.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import distance

    >>> emb_q = np.array([1, 1])

    >>> emb_documents = [
    ...     np.array([0, 10]),
    ...     np.array([1, 1]),
    ... ]

    >>> print(distance.dot_similarity(emb_q=emb_q, emb_documents=emb_documents))
    [(0, 10), (1, 2)]

    """
    distances = {}
    for index, emb_document in enumerate(emb_documents):
        distances[index] = emb_q @ emb_document
    return [
        (index, distance)
        for index, distance in sorted(distances.items(), key=lambda item: item[1], reverse=True)
    ]
