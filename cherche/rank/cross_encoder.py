__all__ = ["CrossEncoder"]

import collections
import typing

import numpy as np

from ..compose import Intersection, Pipeline, Union, Vote
from ..utils import yield_batch


class CrossEncoder:
    """Cross-Encoder as a ranker. CrossEncoder takes both the query and the document as input
    and outputs a score. The score is a similarity score between the query and the document. The CrossEncoder cannot pre-compute the embeddings of the documents since it need both the query
    and the document.

    Parameters
    ----------
    on
        Fields to use to match the query to the documents.
    encoder
        Sentence Transformer cross-encoder.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve, rank, evaluate, data
    >>> from sentence_transformers import CrossEncoder

    >>> documents, query_answers = data.arxiv_tags(
    ...    arxiv_title=True, arxiv_summary=False, comment=False
    ... )

    >>> retriever = retrieve.TfIdf(
    ...    key="uri",
    ...    on=["prefLabel_text", "altLabel_text"],
    ...    documents=documents,
    ...    k=100,
    ... )

    >>> ranker = rank.CrossEncoder(
    ...     on = ["prefLabel_text", "altLabel_text"],
    ...     encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").predict,
    ... )

    >>> pipeline = retriever + documents + ranker

    >>> match = pipeline("graph neural network", k=5)

    >>> for m in match:
    ...     print(m.get("uri", ""))
    'http://www.semanlink.net/tag/graph_neural_networks'
    'http://www.semanlink.net/tag/artificial_neural_network'
    'http://www.semanlink.net/tag/dans_deep_averaging_neural_networks'
    'http://www.semanlink.net/tag/recurrent_neural_network'
    'http://www.semanlink.net/tag/convolutional_neural_network'

    References
    ----------
    1. [Sentence Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
    2. [Cross-Encoders Hub](https://huggingface.co/cross-encoder)

    """

    def __init__(
        self,
        on: typing.Union[typing.List[str], str],
        encoder,
        k: typing.Optional[int] = None,
        batch_size: int = 64,
    ):
        self.on = on if isinstance(on, list) else [on]
        self.encoder = encoder
        self.k = k
        self.batch_size = batch_size

    def __repr__(self) -> str:
        repr = "Cross-encoder"
        repr += f"\n\ton: {', '.join(self.on)}"
        return repr

    def __call__(
        self,
        q: str,
        documents: list,
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        **kwargs,
    ) -> list:
        """Rank inputs documents based on query.

        Parameters
        ----------
        q
            Inputs query.
        documents
            List of documents to rank.

        """
        if k is None:
            k = self.k

        if isinstance(q, str) and not documents:
            return []

        if isinstance(q, list) and not documents:
            return [[]]

        if isinstance(q, str):
            queries = [q]
            documents = [documents]
        else:
            queries = q

        pairs = self._get_pairs(queries=queries, documents=documents)
        scores = self._get_scores(
            pairs=pairs,
            batch_size=batch_size if batch_size is not None else self.batch_size,
        )
        ranked = self._get_rank(documents=documents, scores=scores, k=k)
        return ranked[0] if isinstance(q, str) else ranked

    def _get_pairs(
        self,
        queries: typing.List[str],
        documents: typing.List[typing.List[typing.Dict[str, str]]],
    ) -> typing.List[typing.Tuple[str, str]]:
        """Format input pairs of documents and queries to feed the cross-encoder."""
        pairs = []
        for query, documents_query in zip(queries, documents):
            pairs.extend(
                [
                    (query, " ".join([document.get(field, "") for field in self.on]))
                    for document in documents_query
                ]
            )

        return pairs

    def _get_scores(
        self, pairs: typing.List[typing.Tuple[str, str]], batch_size: int
    ) -> collections.deque:
        """Compute scores for each pair of query and document."""
        scores = []
        for batch in yield_batch(
            array=pairs,
            batch_size=batch_size,
            desc=f"{self.__class__.__name__} ranker",
        ):
            scores.extend(self.encoder(batch))
        return collections.deque(scores)

    def _get_rank(
        self,
        documents: typing.List[typing.List[typing.Dict[str, str]]],
        scores: collections.deque,
        k: int,
    ):
        """Rank documents based on scores."""
        ranked = []

        for n_query, documents_query in enumerate(documents):
            # Extract scores for current query
            array_scores = np.array(
                [scores.popleft() for n_document in range(len(documents_query))]
            ).reshape(1, -1)

            # Sort scores ascending order
            match = np.argsort(array_scores, axis=1)

            # Keep top last k scores
            if k is not None:
                match = match[:, -k:]

            # Reverse order
            match = np.fliplr(match)

            # Extract scores in the same order
            array_scores = np.take_along_axis(array_scores, match, axis=1)

            # Append ranked documents
            ranked.append(
                [
                    {**document, "similarity": similarity}
                    for document, similarity in zip(
                        np.take(documents_query, match.flatten(), axis=-1),
                        array_scores.flatten(),
                    )
                ]
            )

        return ranked

    def __add__(self, other) -> Pipeline:
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return other + self
        else:
            return Pipeline(models=[other, self])

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

    def __mul__(self, other) -> Vote:
        """Voting operator."""
        if isinstance(other, Vote):
            return Vote([self] + other.models)
        return Vote([self, other])
