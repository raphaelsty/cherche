__all__ = ["DPR"]


import typing

from .base import MemoryStore, Ranker


class DPR(Ranker):
    """Dual Sentence Transformer as a ranker. This ranker is compatible with any
    SentenceTransformer. DPR is a dual encoder model, it uses two SentenceTransformer,
    one for encoding documents and one for encoding queries.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields on wich encoder will perform similarity matching.
    encoder
        Encoding function dedicated documents.
    query_encoder
        Encoding function dedicated to queries.
    normalize
        If set to True, the similarity measure is cosine similarity, if set to False, similarity
        measure is dot product.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> ranker = rank.DPR(
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
    ...    normalize = True,
    ... )

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> ranker.add(documents=documents)
    DPR ranker
        key       : id
        on        : title, article
        normalize : True
        embeddings: 3

    >>> match = ranker(
    ...     q="Paris",
    ...     documents=documents
    ... )

    >>> print(match)
    [{'id': 0, 'similarity': 7.806636, 'title': 'Paris France'},
     {'id': 1, 'similarity': 6.239272, 'title': 'Madrid Spain'},
     {'id': 2, 'similarity': 6.168748, 'title': 'Montreal Canada'}]

    >>> match = ranker(
    ...     q=["Paris", "Madrid"],
    ...     documents=[documents + [{"id": 3, "title": "Paris"}]] * 2,
    ...     k=2,
    ... )

    >>> print(match)
    [[{'id': 3, 'similarity': 7.906666, 'title': 'Paris'},
      {'id': 0, 'similarity': 7.806636, 'title': 'Paris France'}],
     [{'id': 1, 'similarity': 8.07025, 'title': 'Madrid Spain'},
      {'id': 0, 'similarity': 6.1131663, 'title': 'Paris France'}]]

    """

    def __init__(
        self,
        on: typing.Union[str, typing.List[str]],
        key: str,
        encoder,
        query_encoder,
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
        self.query_encoder = query_encoder

    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        k: int = None,
        batch_size: typing.Optional[int] = None,
        **kwargs,
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
            embeddings_queries=self.query_encoder([q] if isinstance(q, str) else q),
            documents=[documents] if isinstance(q, str) else documents,
            k=k,
            batch_size=batch_size if batch_size is not None else self.batch_size,
        )

        return rank[0] if isinstance(q, str) else rank
