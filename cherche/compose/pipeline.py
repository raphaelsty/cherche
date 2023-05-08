__all__ = ["Pipeline"]

import collections
import typing

from scipy.special import softmax

from .base import Compose, rank_intersection, rank_union, rank_vote


class PipelineUnion(Compose):
    """Pipeline union.

    Parameters
    ----------
    models
        List of models to include into the union.

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

    >>> retriever_tfidf = retrieve.TfIdf(
    ...     key="id", on=["town"], documents=documents)

    >>> retriever_flash = retrieve.Flash(
    ...     key="id", on=["continent"])

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["town", "country", "continent"],
    ... )

    >>> pipeline = (retriever_tfidf + documents) | (retriever_flash + documents)

    >>> pipeline = pipeline.add(documents)

    >>> print(pipeline("Paris Europe North America"))
    [{'continent': 'Europe',
      'country': 'France',
      'id': 0,
      'similarity': 3.0,
      'town': 'Paris'},
     {'continent': 'Europe',
      'country': 'Spain',
      'id': 2,
      'similarity': 0.3333333333333333,
      'town': 'Madrid'},
     {'continent': 'North America',
      'country': 'Canada',
      'id': 1,
      'similarity': 0.25,
      'town': 'Montreal'}]

    >>> print(pipeline(["Paris Europe", "Madrid Europe"]))
    [[{'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 3.0,
       'town': 'Paris'},
      {'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 0.3333333333333333,
       'town': 'Madrid'}],
     [{'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 2.6666666666666665,
       'town': 'Madrid'},
      {'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 0.5,
       'town': 'Paris'}]]

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Union Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

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
        query = self._build_query(
            q=q, batch_size=batch_size, k=k, documents=documents, **kwargs
        )
        match = self._build_match(query=query)
        scores, _ = self._scores(match=match)
        ranked = rank_union(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "PipelineUnion":
        return PipelineUnion(models=self.models + [other])

    def __and__(self, other) -> "PipelineIntersection":
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> "PipelineVote":
        """Custom operator for voting."""
        return PipelineVote(models=[self, other])

    def __add__(self, other) -> "Pipeline":
        """Pipeline operator."""
        if isinstance(other, list):
            # Mapping to documents
            return Pipeline(
                models=[self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[self, other])


class PipelineIntersection(Compose):
    """Pipelines intersection

    Parameters
    ----------
    model
        List of Pipelines of the union.

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

    >>> retriever_tfidf = retrieve.TfIdf(
    ...     key="id", on=["town"], documents=documents)

    >>> retriever_flash = retrieve.Flash(
    ...     key="id", on=["continent"])

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["town", "country", "continent"],
    ... )

    >>> pipeline = (retriever_tfidf + documents) & (retriever_flash + documents)

    >>> pipeline = pipeline.add(documents)

    >>> print(pipeline("Paris Europe"))
    [{'continent': 'Europe',
      'country': 'France',
      'id': 0,
      'similarity': 3.0,
      'town': 'Paris'}]

    >>> print(pipeline(["Paris Europe", "Madrid Europe"]))
    [[{'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 3.0,
       'town': 'Paris'}],
     [{'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 2.6666666666666665,
       'town': 'Madrid'}]]

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Intersection Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __call__(
        self,
        q: typing.Union[
            typing.List[str],
            str,
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
        scores, counter = self._scores(match=match)
        ranked = rank_intersection(
            key=self.key,
            models=self.models,
            match=match,
            scores=scores,
            counter=counter,
        )
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "PipelineUnion":
        """Custom operator for union."""
        return PipelineUnion(models=[self, other])

    def __and__(self, model) -> "PipelineIntersection":
        return PipelineIntersection(models=self.models + [model])

    def __mul__(self, other) -> "PipelineVote":
        """Custom operator for voting."""
        return PipelineVote(models=[self, other])

    def __add__(self, other) -> "Pipeline":
        """Pipeline operator."""
        if isinstance(other, list):
            # Mapping to documents.
            return Pipeline(
                models=[self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[self, other])


class PipelineVote(Compose):
    """Pipeline voting operator. Average of the similarity scores of the documents between
    pipelines.

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

    >>> retriever_tfidf = retrieve.TfIdf(
    ...     key="id", on=["town", "country", "continent"], documents=documents)

    >>> retriever_flash = retrieve.Flash(
    ...     key="id", on=["town", "country", "continent"])

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["town", "country", "continent"],
    ... )

    >>> pipeline = (retriever_tfidf + documents) * (retriever_flash + documents)

    >>> pipeline = pipeline.add(documents)

    >>> print(pipeline("Paris Europe spain canada", k=3))
    [{'continent': 'Europe',
      'country': 'Spain',
      'id': 2,
      'similarity': 2.4,
      'town': 'Madrid'},
     {'continent': 'Europe',
      'country': 'France',
      'id': 0,
      'similarity': 1.5,
      'town': 'Paris'},
     {'continent': 'North America',
      'country': 'Canada',
      'id': 1,
      'similarity': 1.0,
      'town': 'Montreal'}]

    >>> print(pipeline(["Paris Europe", "Spain Europe"]))
    [[{'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 2.6666666666666665,
       'town': 'Paris'},
      {'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 1.5,
       'town': 'Madrid'}],
     [{'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 2.6666666666666665,
       'town': 'Madrid'},
      {'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 1.5,
       'town': 'Paris'}]]

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Voting Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __call__(
        self,
        q: typing.Union[
            typing.List[str],
            str,
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
        scores, counter = self._scores(match=match)
        ranked = rank_vote(key=self.key, match=match, scores=scores)
        return ranked[0] if isinstance(q, str) else ranked

    def __or__(self, other) -> "PipelineUnion":
        """Custom operator for union."""
        return PipelineUnion(models=[self, other])

    def __and__(self, other) -> "PipelineIntersection":
        """Custom operator for intersection."""
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> "PipelineVote":
        """Custom operator for voting."""
        return PipelineVote(models=self.models + [other])

    def __add__(self, other) -> "Pipeline":
        if isinstance(other, list):
            # Mapping to documents.
            return Pipeline(
                models=[self, {document[self.key]: document for document in other}]
            )
        return Pipeline(models=[self, other])


class Pipeline(Compose):
    """Neurals search pipeline.

    Parameters
    ----------
    models
        List of models of the pipeline.

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

    >>> retriever = retrieve.TfIdf(
    ...     key="id", on=["town", "country", "continent"], documents=documents)

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["town", "country", "continent"],
    ... )

    >>> pipeline = retriever + ranker

    >>> pipeline = pipeline.add(documents)

    >>> print(pipeline("Paris Europe"))
    [{'id': 0, 'similarity': 0.9149576}, {'id': 2, 'similarity': 0.8091332}]

    >>> print(pipeline(["Paris", "Europe", "Paris Madrid Europe France Spain"]))
    [[{'id': 0, 'similarity': 0.69523287}],
     [{'id': 0, 'similarity': 0.7381397}, {'id': 2, 'similarity': 0.6488539}],
     [{'id': 0, 'similarity': 0.8582063}, {'id': 2, 'similarity': 0.8200009}]]

    >>> pipeline = retriever + ranker + documents

    >>> print(pipeline("Paris Europe"))
    [{'continent': 'Europe',
      'country': 'France',
      'id': 0,
      'similarity': 0.9149576,
      'town': 'Paris'},
     {'continent': 'Europe',
      'country': 'Spain',
      'id': 2,
      'similarity': 0.8091332,
      'town': 'Madrid'}]

    >>> print(pipeline(["Paris", "Europe", "Paris Madrid Europe France Spain"]))
    [[{'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 0.69523287,
       'town': 'Paris'}],
     [{'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 0.7381397,
       'town': 'Paris'},
      {'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 0.6488539,
       'town': 'Madrid'}],
     [{'continent': 'Europe',
       'country': 'France',
       'id': 0,
       'similarity': 0.8582063,
       'town': 'Paris'},
      {'continent': 'Europe',
       'country': 'Spain',
       'id': 2,
       'similarity': 0.8200009,
       'town': 'Madrid'}]]

    """

    def __init__(self, models: list) -> None:
        super().__init__(models=models)

    def __call__(
        self,
        q: typing.Union[
            typing.List[str],
            str,
        ],
        k: typing.Optional[int] = None,
        batch_size: typing.Optional[int] = None,
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Pipeline main method. It takes a query and returns a list of documents.
        If the query is a list of queries, it returns a list of list of documents.
        If the batch_size_ranker, or batch_size_retriever it takes precedence over
        the batch_size. If the k_ranker, or k_retriever it takes precedence over
        the k parameter.

        Parameters
        ----------
        q
            Input query.
        k
            Number of documents to return.
        k_retriever
            Number of documents to return for the retriever.
        k_ranker
            Number of documents to return for the ranker.
        batch_size
            Batch size.
        batch_size_retriever
            Batch size for the retriever.
        batch_size_ranker
            Batch size for the ranker.
        """
        query = {**kwargs, "documents": documents}

        for model in self.models:
            if isinstance(model, dict):
                # Map documents to answers.
                answer = self._map_documents(documents=model, answer=query["documents"])
            else:
                # Set k and batch_size.
                answer = model(q=q, k=k, batch_size=batch_size, **query)

            query.update({"documents": answer})

        return query["documents"]

    def _map_documents(
        self,
        documents: typing.Dict[str, typing.Dict[str, str]],
        answer: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Map retrieved documents to the set of documents.

        Parameters
        ----------
        documents
            Mapping of documents in order to map their content on documents ids.
        answer
            List of answers.

        """
        if not answer:
            return answer

        batch = True
        if isinstance(answer[0], dict):
            batch = False
            answer = [answer]

        mapping = []
        for query_document in answer:
            mapping.append(
                [
                    {
                        **documents.get(document[self.key], {}),
                        self.key: document[self.key],
                        "similarity": document["similarity"],
                    }
                    for document in query_document
                ]
            )

        return mapping[0] if not batch else mapping

    def __or__(self, other) -> PipelineUnion:
        """Custom operator for union."""
        return PipelineUnion(models=[self, other])

    def __and__(self, other) -> PipelineIntersection:
        """Custom operator for intersection."""
        return PipelineIntersection(models=[self, other])

    def __mul__(self, other) -> PipelineVote:
        """Custom operator for voting."""
        return PipelineVote(models=[self, other])

    def __add__(self, other) -> "Pipeline":
        """Pipeline operator."""
        if isinstance(other, list):
            # Mapping to documents.
            return Pipeline(
                models=self.models
                + [{document[self.key]: document for document in other}]
            )
        return Pipeline(models=self.models + [other])
