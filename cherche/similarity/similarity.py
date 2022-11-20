__all__ = ["cosine", "dot"]

import typing

import numpy as np


def cosine(
    emb_q: np.ndarray, emb_documents: list, batch: bool = False, k: int = None
) -> typing.Union[list, typing.Tuple[np.ndarray, np.ndarray]]:
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
    if batch:
        array_similarities = np.stack(
            [
                (q @ doc) / (np.linalg.norm(q, axis=0) * np.linalg.norm(doc, axis=0))
                for idx, (q, doc) in enumerate(zip(emb_q, emb_documents))
            ],
            axis=0,
        )
        array_ranks = np.fliplr(np.argsort(array_similarities, axis=1))[:, :k]
        array_similarities = np.take_along_axis(array_similarities, array_ranks, axis=1)
        return (
            array_ranks,
            array_similarities,
        )

    distances = {}
    for index, emb_document in enumerate(emb_documents):
        distances[index] = (emb_q @ emb_document) / (
            np.linalg.norm(emb_q) * np.linalg.norm(emb_document)
        )
    return [
        (index, float(distance))
        for index, distance in sorted(
            distances.items(), key=lambda item: item[1], reverse=True
        )
    ]


def dot(
    emb_q: np.ndarray, emb_documents: list, batch: bool = False, k: int = None
) -> typing.Union[list, typing.Tuple[np.ndarray, np.ndarray]]:
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
    [(0, 10.0), (1, 2.0)]

    """
    if batch:
        array_similarities = np.stack(
            [q @ doc for idx, (q, doc) in enumerate(zip(emb_q, emb_documents))],
            axis=0,
        )
        array_ranks = np.fliplr(np.argsort(array_similarities, axis=1))[:, :k]
        array_similarities = np.take_along_axis(array_similarities, array_ranks, axis=1)
        return (
            array_ranks,
            array_similarities,
        )

    distances = {}
    for index, emb_document in enumerate(emb_documents):
        distances[index] = emb_q @ emb_document
    return [
        (index, float(distance))
        for index, distance in sorted(
            distances.items(), key=lambda item: item[1], reverse=True
        )
    ]
