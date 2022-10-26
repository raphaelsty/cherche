__all__ = ["Pipeline"]

import collections
import typing

from river import stats
from scipy.special import softmax

from .base import Compose


class PipelineUnion(Compose):
    """Pipeline union.

    Parameters
    ----------
    models
        List of models to include into the union.

    Examples
    --------
    >>> from cherche import retrieve
    >>> from pprint import pprint as print

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "author": "Wiki"},
    ... ]

    >>> search = (
    ...     (retrieve.TfIdf(key="id", on="title", documents=documents) + documents) |
    ...     (retrieve.TfIdf(key="id", on="author", documents=documents) + documents)
    ... )

    >>> search
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----

    >>> print(search("paris"))
    [{'author': 'Wiki', 'id': 0, 'similarity': 1.0, 'title': 'Paris'}]

    >>> print(search("wiki"))
    [{'author': 'Wiki',
    'id': 2,
    'similarity': 0.3333333333333333,
    'title': 'Montreal'},
    {'author': 'Wiki',
    'id': 1,
    'similarity': 0.3333333333333333,
    'title': 'Eiffel tower'},
    {'author': 'Wiki',
    'id': 0,
    'similarity': 0.3333333333333333,
    'title': 'Paris'}]


    >>> search + documents
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Mapping to documents


    >>> search * search
    Voting Pipeline
    -----
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    -----

    >>> search & search
    Intersection Pipeline
    -----
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    -----

    >>> search | search
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    Union Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    -----

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
        self, q: str = "", user: typing.Union[str, int] = None, **kwargs
    ) -> list:
        """Query for pipelines union.

        Parameters
        ----------
        q
            Input query.
        user
            Input user.

        """
        query = {"q": q, "user": user, **kwargs}
        union = []
        scores = {}

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
                    scores[document[self.key]] = stats.Mean()
                scores[document[self.key]].update(similarity)

                # Drop duplicates documents:
                if document in union:
                    continue
                union.append(document)

        return [
            {**document, **{"similarity": scores[document[self.key]].get()}}
            for document in union
        ]

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
            # Documents are part of the pipeline.
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
    >>> from cherche import retrieve
    >>> from pprint import pprint as print

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "author": "Wiki"},
    ... ]

    >>> search = (
    ...     (retrieve.TfIdf(key="id", on="title", documents=documents) + documents) &
    ...     (retrieve.TfIdf(key="id", on="author", documents=documents) + documents)
    ... )

    >>> search
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----

    >>> print(search("paris"))
    []

    >>> print(search("wiki paris montreal eiffel"))
    [{'author': 'Wiki',
      'id': 2,
      'similarity': 0.3424492551376574,
      'title': 'Montreal'},
     {'author': 'Wiki',
      'id': 0,
      'similarity': 0.3424492551376574,
      'title': 'Paris'},
     {'author': 'Wiki',
      'id': 1,
      'similarity': 0.3151014897246852,
      'title': 'Eiffel tower'}]

    >>> search + documents
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Mapping to documents

    >>> search * search
    Voting Pipeline
    -----
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    -----

    >>> search | search
    Union Pipeline
    -----
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    -----

    >>> search & search
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    -----

    >>> from cherche import rank
    >>> from sentence_transformers import SentenceTransformer

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title", "article"],
    ...    k = 2,
    ... )

    >>> search + ranker
    Intersection Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: author
         documents: 3
    Mapping to documents
    -----
    Encoder ranker
         key: id
         on: title, article
         k: 2
         similarity: cosine
         Embeddings pre-computed: 3

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
        self, q: str = "", user: typing.Union[str, int] = None, **kwargs
    ) -> list:
        """Retrieve documents.

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
            # Documents are part of the pipeline.
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

    >>> from cherche import compose, retrieve
    >>> from pprint import pprint as print

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> search = (
    ...    (retrieve.TfIdf(key="id", on="title", documents=documents, k=3) + documents) *
    ...    (retrieve.TfIdf(key="id", on="article", documents=documents, k=3) + documents)
    ... )

    >>> search
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    -----

    >>> print(search("paris eiffel"))
    [{'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.5216793798120437,
      'title': 'Eiffel tower'},
     {'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.4783206201879563,
      'title': 'Paris'}]

    >>> search + documents
    Voting Pipeline
    -----
    TfIdf retriever
        key: id
        on: title
        documents: 3
    Mapping to documents
    TfIdf retriever
        key: id
        on: article
        documents: 3
    Mapping to documents
    -----
    Mapping to documents

    >>> search * search
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    -----
    -----

    >>> search & search
    Intersection Pipeline
    -----
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    -----
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    -----
    -----

    >>> search | search
    Union Pipeline
    -----
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    -----
    Voting Pipeline
    -----
    TfIdf retriever
         key: id
         on: title
         documents: 3
    Mapping to documents
    TfIdf retriever
         key: id
         on: article
         documents: 3
    Mapping to documents
    -----
    -----


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
            # Documents are part of the pipeline.
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
    >>> from cherche import retrieve, rank, qa, summary
    >>> from sentence_transformers import SentenceTransformer
    >>> from transformers import pipeline

    >>> documents = [
    ...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
    ...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
    ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.TfIdf(key="id", on="article", documents=documents)

    Retriever, Ranker:
    >>> ranker = rank.Encoder(
    ...    on = "article",
    ...    key = "id",
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    path = "pipeline_encoder.pkl"
    ... )

    >>> search = retriever + ranker + documents

    >>> search.add(documents=documents)
    TfIdf retriever
        key: id
        on: article
        documents: 3
    Encoder ranker
        key: id
        on: article
        k: None
        similarity: cosine
        Embeddings pre-computed: 3
    Mapping to documents

    >>> print(search(q = "Paris"))
    [{'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.7014107704162598,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.5178720951080322,
      'title': 'Eiffel tower'}]

    Retriever, Ranker, Question Answering:
    >>> search += qa.QA(
    ...     model = pipeline(
    ...         "question-answering",
    ...         model = "deepset/roberta-base-squad2",
    ...         tokenizer = "deepset/roberta-base-squad2"
    ...     ),
    ...     on = "article",
    ... )

    >>> search
    TfIdf retriever
        key: id
        on: article
        documents: 3
    Encoder ranker
        key: id
        on: article
        k: None
        similarity: cosine
        Embeddings pre-computed: 3
    Mapping to documents
    Question Answering
        on: article

    >>> print(search(q = "What is based in Paris?"))
    [{'answer': 'Eiffel tower',
      'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'end': 12,
      'id': 1,
      'qa_score': 0.9643093943595886,
      'similarity': 0.6578713655471802,
      'start': 0,
      'title': 'Eiffel tower'},
     {'answer': 'Paris is the capital of France',
      'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'end': 30,
      'id': 0,
      'qa_score': 4.2473871872061864e-05,
      'similarity': 0.7062915563583374,
      'start': 0,
      'title': 'Paris'},
     {'answer': 'Montreal is in Canada.',
      'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'end': 22,
      'id': 2,
      'qa_score': 1.7172791189068448e-08,
      'similarity': 0.3316514492034912,
      'start': 0,
      'title': 'Montreal'}]

    """

    def __init__(self, models: list) -> None:
        super().__init__(models=models)

    def __call__(
        self, q: str = "", user: typing.Union[str, int] = None, **kwargs
    ) -> list:
        """Compose pipeline

        Parameters
        ----------
        q
            Input query.
        user
            Input user.

        """
        query = {"user": user, **kwargs}
        summary = False

        for model in self.models:

            # mapping documents
            if isinstance(model, dict):

                answer = [
                    {**model[doc[self.key]], "similarity": doc["similarity"]}
                    if "similarity" in doc
                    else model[doc[self.key]]
                    for doc in answer
                ]

            # retriever, ranker, qa, summarization
            else:
                answer = model(q=q, **query)

            # retriever, ranker, qa,
            if isinstance(answer, list):
                query.update({"documents": answer})

            # Query translation and summarization
            elif isinstance(answer, str):
                q = answer

                # Translation of the summary may happend.
                if model.type == "summary":
                    summary = True

        return query["documents"] if not summary else q

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
            # Documents are part of the pipeline.
            return Pipeline(
                models=self.models
                + [{document[self.key]: document for document in other}]
            )
        return Pipeline(models=self.models + [other])
