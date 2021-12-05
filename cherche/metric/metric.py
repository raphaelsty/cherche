__all__ = ["cosine_distance", "dot_similarity"]

import numpy as np
from scipy.spatial import distance


def cosine_distance(q: np.ndarray, documents: list):
    """Computes cosine distance (lower is better) between input query embedding and documents
    embeddings.

    Parameters
    ----------

        q: Embedding of the query.
        documents: List of embeddings of the documents.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import metric

    >>> q = np.array([1, 1])

    >>> documents = [
    ...     np.array([0, 10]),
    ...     np.array([1, 1]),
    ... ]

    >>> print(metric.cosine_distance(q=q, documents=documents))
    [(1, 0.0), (0, 0.29289321881345254)]

    """
    distances = {}
    for index, document in enumerate(documents):
        distances[index] = distance.cosine(q, document)
    return [
        (index, distance)
        for index, distance in sorted(distances.items(), key=lambda item: item[1])
    ]


def dot_similarity(q: np.ndarray, documents: list):
    """Computes dot product (higher is better) between input query embedding and documents
    embeddings

    Parameters
    ----------

        q: Embedding of the query.
        documents: List of embeddings of the documents.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import metric

    >>> q = np.array([1, 1])

    >>> documents = [
    ...     np.array([0, 10]),
    ...     np.array([1, 1]),
    ... ]

    >>> print(metric.dot_similarity(q=q, documents=documents))
    [(0, 10), (1, 2)]

    """
    distances = {}
    for index, document in enumerate(documents):
        distances[index] = q @ document
    return [
        (index, distance)
        for index, distance in sorted(distances.items(), key=lambda item: item[1], reverse=True)
    ]
