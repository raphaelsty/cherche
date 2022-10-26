import typing

from scipy import sparse

__all__ = ["users_items_sparse"]


def users_items_sparse(ratings: dict) -> typing.Tuple[list, list, sparse.csr_matrix]:
    """Convert dict to scipy csr matrix for Implicit.

    Parameters
    ----------
    ratings
        Dict with users as key and documents with their evaluations as values.

    Examples
    --------
    >>> from cherche import utils

    >>> ratings = {
    ...    "Max": {"a": 1, "c": 1},
    ...    "Adil": {"b": 1, "d": 2},
    ...    "Robin": {"b": 1, "d": 1},
    ...    "Geoffrey": {"a": 1, "c": 1},
    ... }

    >>> users, documents, matrix = utils.users_items_sparse(ratings=ratings)

    >>> users
    ['Max', 'Adil', 'Robin', 'Geoffrey']

    >>> documents
    ['a', 'c', 'b', 'd']

    >>> matrix
    <4x4 sparse matrix of type '<class 'numpy.int64'>'
        with 8 stored elements in Compressed Sparse Row format>

    """
    indptr, indices, data, users = [0], [], [], []
    index: typing.Dict[int, int] = {}

    for user_id, likes in ratings.items():
        for doc_id, like in likes.items():
            indices.append(index.setdefault(doc_id, len(index)))
            data.append(like)
        indptr.append(len(data))
        users.append(user_id)

    return users, [key for key in index], sparse.csr_matrix((data, indices, indptr))
