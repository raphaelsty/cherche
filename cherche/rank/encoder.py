__all__ = ["Encoder"]

import typing

from .base import MemoryStore, Ranker


class Encoder(Ranker):
    """Sentence Transformer as a ranker. This ranker is compatible with any SentenceTransformer.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields on wich encoder will perform similarity matching.
    encoder
        Encoding function dedicated to both documents and queries.
    normalize
        If set to True, the similarity measure is cosine similarity, if set to False, similarity
        measure is dot product.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title"],
    ... )

    >>> ranker.add(documents=documents)
    Encoder ranker
        key       : id
        on        : title
        normalize : True
        embeddings: 3

    >>> match = ranker(
    ...     q="Paris",
    ...     documents=documents
    ... )

    >>> print(match)
    [{'id': 0, 'similarity': 0.7127624, 'title': 'Paris France'},
     {'id': 1, 'similarity': 0.5497405, 'title': 'Madrid Spain'},
     {'id': 2, 'similarity': 0.50252455, 'title': 'Montreal Canada'}]

    >>> match = ranker(
    ...     q=["Paris France", "Madrid Spain"],
    ...     documents=[documents + [{"id": 3, "title": "Paris"}]] * 2,
    ...     k=2,
    ... )

    >>> print(match)
    [[{'id': 0, 'similarity': 0.99999994, 'title': 'Paris France'},
      {'id': 1, 'similarity': 0.856435, 'title': 'Madrid Spain'}],
     [{'id': 1, 'similarity': 1.0, 'title': 'Madrid Spain'},
      {'id': 0, 'similarity': 0.856435, 'title': 'Paris France'}]]

    """

    def __init__(
        self,
        on: typing.Union[str, typing.List[str]],
        key: str,
        encoder,
        normalize: bool = True,
        k: typing.Optional[int] = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            key=key,
            on=on,
            encoder=encoder,
            normalize=normalize,
            k=k,
            batch_size=batch_size,
        )

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        **kwargs
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Encode input query and ranks documents based on the similarity between the query and
        the selected field of the documents.

        Parameters
        ----------
        q
            Input query.
        documents
            List of documents to rank.

        """
        if k is None:
            k = self.k

        if k is None:
            k = len(self)

        if not documents and isinstance(q, str):
            return []

        if not documents and isinstance(q, list):
            return [[]]

        rank = self.encode_rank(
            embeddings_queries=self.encoder([q] if isinstance(q, str) else q),
            documents=[documents] if isinstance(q, str) else documents,
            k=k,
            batch_size=batch_size if batch_size is not None else self.batch_size,
        )

        return rank[0] if isinstance(q, str) else rank
