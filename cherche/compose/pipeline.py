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
        self.models = models

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
        self.models.append(model)
        return self


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
        """
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
        self.models.append(model)
        return self


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
    ...     {"title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
    ...     {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
    ...     {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> retriever = retrieve.TfIdf(on="article")

    Retriever, Ranker:
    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    on = "article",
    ...    path = "pipeline_encoder.pkl"
    ... )

    >>> search = retriever + ranker

    >>> search.add(documents=documents)
    TfIdf retriever
         on: article
         documents: 3
    Encoder ranker
         on: article
         k: None
         similarity: cosine
         embeddings stored at: pipeline_encoder.pkl

    >>> print(search(q = "Paris"))
    [{'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'similarity': 0.7014109,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'similarity': 0.51787204,
      'title': 'Eiffel tower'}]

    Retriever, Ranker, Question Answering:
    >>> search += qa.QA(
    ...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
    ...     on = "article",
    ...  )

    >>> search
    TfIdf retriever
         on: article
         documents: 3
    Encoder ranker
         on: article
         k: None
         similarity: cosine
         embeddings stored at: pipeline_encoder.pkl
    Question Answering
         model: deepset/roberta-base-squad2
         on: article

    >>> print(search(q = "What is based in Paris?"))
    [{'answer': 'Eiffel tower',
      'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'end': 12,
      'qa_score': 0.9643093347549438,
      'similarity': 0.65787125,
      'start': 0,
      'title': 'Eiffel tower'},
     {'answer': 'Paris is the capital of France',
      'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'end': 30,
      'qa_score': 4.247476681484841e-05,
      'similarity': 0.7062913,
      'start': 0,
      'title': 'Paris'},
     {'answer': 'Montreal is in Canada.',
      'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'end': 22,
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
        for model in self.models:
            answer = model(q=q, **query)
            if isinstance(answer, list):
                query.update({"documents": answer})
            elif isinstance(answer, str):
                return answer
        return query["documents"]

    def __or__(self, other) -> PipelineUnion:
        """Custom operator for union."""
        if isinstance(other, PipelineUnion):
            return PipelineUnion(models=[self] + [other.models])
        else:
            return PipelineUnion(models=[self, other])

    def __and__(self, other) -> PipelineIntersection:
        """Custom operator for intersection."""
        if isinstance(other, PipelineIntersection):
            return PipelineIntersection(models=[self] + [other.models])
        else:
            return PipelineIntersection(models=[self, other])
