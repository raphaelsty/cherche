__all__ = ["Intersection", "Union"]
import collections

from .base import Compose
from .pipeline import Pipeline


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
                [self] + [{document[self.models[0].key]: document for document in other}]
            )
        return Pipeline([self, other])


class Union(UnionIntersection):
    """Union gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers.

    Parameters
    ----------
    model
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
      'similarity': 1.0,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.4505,
      'title': 'Eiffel tower'}]

    >>> print(search(q = "Montreal"))
    [{'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'similarity': 1.0,
      'title': 'Montreal'}]

    >>> print(search(q = "Wiki"))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'title': 'Eiffel tower'},
     {'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'title': 'Montreal'}]

    """

    def __init__(self, models: list):
        super().__init__(models=models)
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    def __call__(self, q: str, **kwargs) -> list:
        """
        Parameters
        ----------
        q
            Input query.

        """
        query = {"q": q, **kwargs}
        union = []
        similarities = {}

        for model in self.models:

            for document in model(**query):

                # Remove similarities to avoid duplicates
                if "similarity" in document:
                    similarity = document.pop("similarity")

                    if document[self.key] not in similarities:
                        similarities[document[self.key]] = similarity

                # Drop duplicates documents:
                if document in union:
                    continue
                union.append(document)

        ranked = []
        for document in union:
            similarity = similarities.get(document.get(self.key, None), None)
            if similarity is not None:
                ranked.append({**document, **{"similarity": similarity}})
            else:
                ranked.append(document)

        return ranked

    def __or__(self, other) -> "Union":
        """Union operator"""
        return Union(models=self.models + [other])


class Intersection(UnionIntersection):
    """Intersection gathers retrieved documents from multiples retrievers and ranked documents from
    multiples rankers only if they are proposed by all models of the intersection pipeline.

    Parameters
    ----------
    model
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
      'similarity': 1.0,
      'title': 'Paris'}]

    >>> print(search(q = "Paris"))
    []

    >>> print(search(q = "Wiki Paris Montreal Eiffel"))
    [{'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.57735,
      'title': 'Paris'},
     {'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'similarity': 0.57735,
      'title': 'Montreal'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.40825,
      'title': 'Eiffel tower'}]

    """

    def __init__(self, models: list):
        super().__init__(models=models)
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    def __call__(self, q: str, **kwargs) -> list:
        """
        Parameters
        ----------
        q
            Input query.

        """
        query = {"q": q, **kwargs}
        similarities = {}
        counter_docs = collections.defaultdict(int)

        for model in self.models:

            for document in model(**query):

                # Remove similarities to avoid duplicates
                if "similarity" in document:
                    similarity = document.pop("similarity")

                    if document[self.key] not in similarities:
                        similarities[document[self.key]] = similarity

                counter_docs[tuple(sorted(document.items()))] += 1

        intersection = [
            dict(document) for document, count in counter_docs.items() if count >= len(self.models)
        ]

        # Add similarity that we previously removed.
        ranked = []
        for document in intersection:
            similarity = similarities.get(document.get(self.key, None), None)
            if similarity is not None:
                ranked.append({**document, **{"similarity": similarity}})
            else:
                ranked.append(document)
        return ranked

    def __and__(self, other) -> "Intersection":
        return Intersection(models=self.models + [other])
