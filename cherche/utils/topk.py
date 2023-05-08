__all__ = ["TopK"]

import typing


class TopK:
    """Filter top k documents in pipeline.

    Parameters
    ----------
    k
        Number of documents to keep.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve, rank, utils
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title"],
    ... )

    >>> pipeline = retriever + ranker + utils.TopK(k=2)
    >>> pipeline.add(documents=documents)
    TfIdf retriever
        key      : id
        on       : title, article
        documents: 3
    Encoder ranker
        key       : id
        on        : title
        normalize : True
        embeddings: 3
    Filter TopK
        k: 2

    >>> print(pipeline(q="Paris Madrid Montreal", k=2))
    [{'id': 0, 'similarity': 0.62922895}, {'id': 2, 'similarity': 0.61419094}]

    """

    def __init__(self, k: int):
        self.k = k

    def __repr__(self) -> str:
        repr = f"Filter {self.__class__.__name__}"
        repr += f"\n\tk: {self.k}"
        return repr

    def __call__(
        self,
        documents: typing.Union[typing.List[typing.List[typing.Dict[str, str]]]],
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Filter top k documents in pipeline."""
        if not documents:
            return []

        if isinstance(documents[0], list):
            return [document[: self.k] for document in documents]

        return documents[: self.k]
