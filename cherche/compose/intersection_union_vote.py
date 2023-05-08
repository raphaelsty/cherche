import collections
import typing

from scipy.special import softmax

from .base import Compose, rank_intersection, rank_union, rank_vote
from .pipeline import Pipeline, PipelineIntersection, PipelineUnion, PipelineVote

__all__ = ["Intersection", "Union", "Vote"]


class IntersectionUnionVote(Compose):
    """Base class for union and intersection."""

    def __repr__(self) -> str:
        repr = self.__class__.__name__
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, list):
            return Pipeline(
                [self, {document[self.models[0].key]: document for document in other}]
            )
        return Pipeline([self, other])

    def __or__(self, other) -> PipelineUnion:
        return PipelineUnion(models=[self, other])

    def __and__(self, other) -> PipelineIntersection:
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> PipelineVote:
        """Custom operator for voting."""
        return PipelineVote(models=[self, other])


class Union(IntersectionUnionVote):
    """Union gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers. The union operator concat results with respect of the orders of the models
    in the union.

    Parameters
    ----------
    models
        List of models of the union.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...    {"id": 0, "town": "Paris", "country": "France", "continent": "Europe"},
    ...    {"id": 1, "town": "Montreal", "country": "Canada", "continent": "North America"},
    ...    {"id": 2, "town": "Madrid", "country": "Spain", "continent": "Europe"},
    ... ]

    >>> search = (
    ...     retrieve.TfIdf(key="id", on="town", documents=documents) |
    ...     retrieve.TfIdf(key="id", on="country", documents=documents) |
    ...     retrieve.Flash(key="id", on="continent")
    ... )

    >>> search = search.add(documents)

    >>> print(search("Paris"))
    [{'id': 0, 'similarity': 1.0}]

    >>> print(search(["Paris", "Europe"]))
    [[{'id': 0, 'similarity': 1.0}],
    [{'id': 0, 'similarity': 1.0}, {'id': 2, 'similarity': 0.5}]]

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """
        Parameters
        ----------
        q
            Input query.

        """
        query = self._build_query(q=q, batch_size=batch_size, k=k, documents=documents)
        match = self._build_match(query=query)
        scores, _ = self._scores(match=match)
        ranked = rank_union(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "Union":
        """Union operator"""
        return Union(models=self.models + [other])


class Intersection(IntersectionUnionVote):
    """Intersection gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers only if they are proposed by all models of the intersection pipeline.

    Parameters
    ----------
    models
        List of models of the union.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...    {"id": 0, "town": "Paris", "country": "France", "continent": "Europe"},
    ...    {"id": 1, "town": "Montreal", "country": "Canada", "continent": "North America"},
    ...    {"id": 2, "town": "Madrid", "country": "Spain", "continent": "Europe"},
    ... ]

    >>> search = (
    ...     retrieve.TfIdf(key="id", on="town", documents=documents) &
    ...     retrieve.TfIdf(key="id", on="country", documents=documents) &
    ...     retrieve.Flash(key="id", on="continent")
    ... )

    >>> search = search.add(documents)

    >>> print(search("Paris"))
    []

    >>> print(search(["Paris", "Europe"]))
    [[], []]

    >>> print(search(["Paris", "Europe", "Paris Madrid Europe France Spain"]))
    [[],
    [],
    [{'id': 2, 'similarity': 4.25}, {'id': 0, 'similarity': 3.0999999999999996}]]

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        query = self._build_query(q=q, batch_size=batch_size, k=k, documents=documents)
        match = self._build_match(query=query)
        scores, counter = self._scores(match=match)
        ranked = rank_intersection(
            key=self.key,
            models=self.models,
            match=match,
            scores=scores,
            counter=counter,
        )
        return ranked[0] if isinstance(q, str) else ranked

    def __and__(self, other) -> "Intersection":
        return Intersection(models=self.models + [other])


class Vote(IntersectionUnionVote):
    """Voting operator. Computes the score for each document based on it's number of occurences
    and based on documents ranks: $nb_occurences * sum_{rank \in ranks} 1 / rank$. The higher the
    score, the higher the document is ranked in output of the vote.

    Parameters
    ----------
    models
        List of models of the vote.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve, rank
    >>> from sentence_transformers import SentenceTransformer


    >>> documents = [
    ...    {"id": 0, "town": "Paris", "country": "France", "continent": "Europe"},
    ...    {"id": 1, "town": "Montreal", "country": "Canada", "continent": "North America"},
    ...    {"id": 2, "town": "Madrid", "country": "Spain", "continent": "Europe"},
    ... ]

    >>> search = (
    ...     retrieve.TfIdf(key="id", on="town", documents=documents) *
    ...     retrieve.TfIdf(key="id", on="country", documents=documents) *
    ...     retrieve.Flash(key="id", on="continent")
    ... )

    >>> search = search.add(documents)

    >>> retriever = retrieve.TfIdf(key="id", on=["town", "country", "continent"], documents=documents)

    >>> ranker = rank.Encoder(
    ...     key="id",
    ...     on=["town", "country", "continent"],
    ...     encoder=SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ... ) * rank.Encoder(
    ...     key="id",
    ...     on=["town", "country", "continent"],
    ...     encoder=SentenceTransformer(
    ...        "sentence-transformers/multi-qa-mpnet-base-cos-v1"
    ...     ).encode,
    ... )

    >>> search = retriever + ranker

    >>> search = search.add(documents)

    >>> print(search("What is the capital of Canada ? Is it paris, montreal or madrid ?"))
    [{'id': 1, 'similarity': 2.5},
     {'id': 0, 'similarity': 1.4},
     {'id': 2, 'similarity': 1.0}]

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        k: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        query = self._build_query(
            q=q, batch_size=batch_size, k=k, documents=documents, **kwargs
        )
        match = self._build_match(query=query)
        scores, _ = self._scores(match=match)
        ranked = rank_vote(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __mul__(self, other) -> "Vote":
        return Vote(models=self.models + [other])
