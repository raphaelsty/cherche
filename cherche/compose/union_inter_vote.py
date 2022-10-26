__all__ = ["Intersection", "Union", "Vote"]
import collections
import typing

from scipy.special import softmax

from .base import Compose
from .pipeline import Pipeline, PipelineIntersection, PipelineUnion, PipelineVote


class UnionIntersection(Compose):
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


class Union(UnionIntersection):
    """Union gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers.

    Parameters
    ----------
    models
        List of models of the union.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> search = (
    ...     retrieve.TfIdf(key="id", on="title", documents=documents) |
    ...     retrieve.TfIdf(key="id", on="article", documents=documents) |
    ...     retrieve.Flash(key="id", on="author")
    ... ) + documents

    >>> search.add(documents)
    Union
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Flash retriever
         key: id
         on: author
         documents: 1
    -----
    Mapping to documents

    >>> print(search(q = "Paris"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.3333333333333333,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.3333333333333333,
      'title': 'Eiffel tower'}]

    >>> print(search(q = "Montreal"))
    [{'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'similarity': 0.3333333333333333,
      'title': 'Montreal'}]

    >>> print(search(q = "Wiki"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.1111111111111111,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.1111111111111111,
      'title': 'Eiffel tower'},
     {'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'similarity': 0.1111111111111111,
      'title': 'Montreal'}]

    """

    def __init__(self, models: list):
        super().__init__(models=models)
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    def __call__(
        self, q: str = "", user: typing.Union[str, int] = None, **kwargs
    ) -> list:
        """
        Parameters
        ----------
        q
            Input query.
        user
            Input user.

        """
        union = []
        query = {"q": q, "user": user, **kwargs}
        scores = collections.defaultdict(float)

        for model in self.models:

            retrieved = model(**query)
            if not retrieved:
                continue

            similarities = softmax(
                [doc.get("similarity", 1.0) for doc in retrieved], axis=0
            )
            for document, similarity in zip(retrieved, similarities):

                # Remove similarities to avoid duplicates
                if "similarity" in document:
                    document.pop("similarity")

                if document[self.key] not in scores:
                    scores[document[self.key]] += float(similarity) / len(self.models)

                # Drop duplicates documents:
                if document in union:
                    continue
                union.append(document)

        return [
            {**document, **{"similarity": scores[document[self.key]]}}
            for document in union
        ]

    def __or__(self, other) -> "Union":
        """Union operator"""
        return Union(models=self.models + [other])


class Intersection(UnionIntersection):
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
    ...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> search = (
    ...    retrieve.TfIdf(key="id", on="title", documents=documents) &
    ...    retrieve.TfIdf(key="id", on="article", documents=documents) &
    ...    retrieve.Flash(key="id", on="author")
    ... ) + documents

    >>> search.add(documents)
    Intersection
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Flash retriever
         key: id
         on: author
         documents: 1
    -----
    Mapping to documents

    >>> print(search(q = "Wiki Paris"))
    [{'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.6098051865303775,
      'title': 'Paris'}]


    >>> print(search(q = "Paris"))
    []

    >>> print(search(q = "Wiki Paris Montreal Eiffel"))
    [{'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'similarity': 0.3423964772770161,
      'title': 'Montreal'},
     {'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.3215535643422896,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.33604995838069424,
      'title': 'Eiffel tower'}]


    """

    def __init__(self, models: list):
        super().__init__(models=models)
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    def __call__(
        self, q: str = "", user: typing.Union[str, int] = None, **kwargs
    ) -> list:
        """
        Parameters
        ----------
        q
            Input query.
        user
            Input user.

        """
        query = {"q": q, "user": user, **kwargs}
        counter_docs, scores = collections.defaultdict(int), collections.defaultdict(
            float
        )

        for model in self.models:
            retrieved = model(**query)
            if not retrieved:
                continue

            similarities = softmax(
                [doc.get("similarity", 1.0) for doc in retrieved], axis=0
            )
            for document, similarity in zip(retrieved, similarities):

                if "similarity" in document:
                    document.pop("similarity")

                scores[document[self.key]] += similarity / len(self.models)
                counter_docs[tuple(sorted(document.items()))] += 1

        ranked = []
        for document, count in counter_docs.items():
            if count < len(self.models):
                continue
            document = dict(document)
            document["similarity"] = scores[document[self.key]]
            ranked.append(document)
        return ranked

    def __and__(self, other) -> "Intersection":
        return Intersection(models=self.models + [other])


class Vote(UnionIntersection):
    """Voting operator. Average of the similarity scores of the documents.

    Parameters
    ----------
    models
        List of models of the vote.

    Examples
    --------

    >>> from cherche import compose, retrieve
    >>> from pprint import pprint as print

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> search = (
    ...    retrieve.TfIdf(key="id", on="title", documents=documents, k=3) *
    ...    retrieve.TfIdf(key="id", on="article", documents=documents, k=3)
    ... )

    >>> search
    Vote
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    TfIdf retriever
         key: id
         on: article
         documents: 3
    -----

    >>> print(search("paris eiffel"))
    [{'id': 1, 'similarity': 0.5216793798120437},
     {'id': 0, 'similarity': 0.4783206201879563}]


    """

    def __init__(self, models: list):
        super().__init__(models=models)
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    def __call__(
        self, q: str = "", user: typing.Union[str, int] = None, **kwargs
    ) -> list:
        """
        Parameters
        ----------
        q
            Input query.
        user
            Input user.

        """
        query = {"q": q, "user": user, **kwargs}

        scores, documents = collections.defaultdict(float), collections.defaultdict(
            dict
        )

        for model in self.models:
            retrieved = model(**query)

            if not retrieved:
                continue

            similarities = softmax(
                [doc.get("similarity", 1.0) for doc in retrieved], axis=0
            )
            for doc, similarity in zip(retrieved, similarities):
                scores[doc[self.key]] += float(similarity) / len(self.models)
                documents[doc[self.key]] = doc

        ranked = []

        if not scores or not documents:
            return []

        for key, similarity in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        ):
            doc = documents[key]
            doc["similarity"] = similarity
            ranked.append(doc)

        return ranked

    def __mul__(self, other) -> "Vote":
        return Vote(models=self.models + [other])
