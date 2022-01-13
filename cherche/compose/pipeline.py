__all__ = ["Pipeline"]

import collections

from .base import Compose


class PipelineUnion(Compose):
    """Pipeline union.

    Parameters
    ----------
    models
        List of models to include into the union.

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Union Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __call__(self, q: str, **kwargs) -> list:
        """
        Parameters
        ----------
        q
            Input query.

        """
        query = {"q": q, **kwargs}
        documents = []
        for model in self.models:
            for document in model(**query):
                if "similarity" in document:
                    document.pop("similarity")
                # Drop duplicates documents:
                if document in documents:
                    continue
                documents.append(document)
        return documents

    def __or__(self, model) -> "PipelineUnion":
        return PipelineUnion(models=self.models + [model])

    def __add__(self, other) -> "PipelineUnion":
        """Pipeline operator."""
        if isinstance(other, list):
            # Documents are part of the pipeline.
            return Pipeline(models=[self, {document[self.key]: document for document in other}])
        return Pipeline(models=[self, other])


class PipelineIntersection(Compose):
    """Pipelines intersection

    Parameters
    ----------
    model
        List of Pipelines of the union.

    """

    def __init__(self, models: list):
        super().__init__(models=models)

    def __repr__(self) -> str:
        repr = "Union Pipeline"
        repr += "\n-----\n"
        repr += super().__repr__()
        repr += "\n-----"
        return repr

    def __call__(self, q: str, **kwargs) -> list:
        """Retrieve documents.

        Parameters
        ----------
        q
            Input query.

        """
        query = {"q": q, **kwargs}
        counter_docs = collections.defaultdict(int)
        for model in self.models:
            for document in model(**query):
                if "similarity" in document:
                    document.pop("similarity")
                counter_docs[tuple(sorted(document.items()))] += 1
        return [
            dict(document) for document, count in counter_docs.items() if count >= len(self.models)
        ]

    def __and__(self, model) -> "PipelineIntersection":
        return PipelineIntersection(models=self.models + [model])

    def __add__(self, other) -> "PipelineIntersection":
        """Pipeline intersection operator."""
        if isinstance(other, list):
            # Documents are part of the pipeline.
            return PipelineIntersection(
                models=self.models + [{document[self.key]: document for document in other}]
            )
        return PipelineIntersection(models=self.models + [other])


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
         embeddings stored at: pipeline_encoder.pkl
    Mapping to documents

    >>> print(search(q = "Paris"))
    [{'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 0.7014109,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'similarity': 0.51787204,
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
         embeddings stored at: pipeline_encoder.pkl
    Mapping to documents
    Question Answering
         model: deepset/roberta-base-squad2
         on: article

    >>> print(search(q = "What is based in Paris?"))
    [{'answer': 'Eiffel tower',
      'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'end': 12,
      'id': 1,
      'qa_score': 0.9643093347549438,
      'similarity': 0.65787125,
      'start': 0,
      'title': 'Eiffel tower'},
     {'answer': 'Paris is the capital of France',
      'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'end': 30,
      'id': 0,
      'qa_score': 4.247476681484841e-05,
      'similarity': 0.7062913,
      'start': 0,
      'title': 'Paris'},
     {'answer': 'Montreal is in Canada.',
      'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'end': 22,
      'id': 2,
      'qa_score': 1.7172554933608808e-08,
      'similarity': 0.3316515,
      'start': 0,
      'title': 'Montreal'}]

    """

    def __init__(self, models: list) -> None:
        super().__init__(models=models)

    def __call__(self, q: str, **kwargs) -> list:
        """Compose pipeline

        Parameters
        ----------
        q
            Input query.

        """
        query = {**kwargs}
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

    def __add__(self, other) -> "Pipeline":
        """Pipeline operator."""
        if isinstance(other, list):
            # Documents are part of the pipeline.
            return Pipeline(
                models=self.models + [{document[self.key]: document for document in other}]
            )
        return Pipeline(models=self.models + [other])
