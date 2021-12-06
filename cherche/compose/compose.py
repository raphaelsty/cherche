__all__ = ["Compose"]


class Compose:
    """Cherche pipeline.

    Parameters
    ----------

        models: List of models to use in the pipeline.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve, rank, qa, summary
    >>> from sentence_transformers import SentenceTransformer
    >>> from transformers import pipeline

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
    ... ]

    >>> retriever = retrieve.TfIdf(on="title")

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    on = "title",
    ...    path = "pipeline_encoder.pkl"
    ... )

    Retriever, Ranker:
    >>> search = retriever + ranker

    >>> search = search.add(documents=documents)

    >>> search
    TfIdf retriever
         on: title
         documents: 3
    Encoder ranker
         on: title
         k: None
         Metric: cosine_distance
         Embeddings stored at: pipeline_encoder.pkl

    >>> print(search(q = "Transformers"))
    [{'cosine_distance': 0.6396294832229614,
      'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers.',
      'url': 'ckb/github.com'},
     {'cosine_distance': 0.6396294832229614,
      'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers.',
      'url': 'blp/github.com'},
     {'cosine_distance': 0.8993543088436127,
      'date': '22-11-2021',
      'title': 'Github Library with PyTorch.',
      'url': 'mkb/github.com'}]

    Retriever, Ranker, Question Answering:
    >>> search += qa.QA(
    ...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
    ...     on = "title",
    ...  )

    >>> search
    TfIdf retriever
         on: title
         documents: 3
    Encoder ranker
         on: title
         k: None
         Metric: cosine_distance
         Embeddings stored at: pipeline_encoder.pkl
    Question Answering
         model: deepset/roberta-base-squad2
         on: title

    >>> print(search(q = "Transformers"))
    [{'answer': 'Github Library with Pytorch and Transformers.',
      'cosine_distance': 0.6396294832229614,
      'date': '22-11-2020',
      'end': 45,
      'qa_score': 1.2880965186923277e-05,
      'start': 0,
      'title': 'Github Library with Pytorch and Transformers.',
      'url': 'blp/github.com'},
     {'answer': 'Github library with PyTorch and Transformers.',
      'cosine_distance': 0.6396294832229614,
      'date': '10-11-2021',
      'end': 45,
      'qa_score': 8.694544703757856e-06,
      'start': 0,
      'title': 'Github library with PyTorch and Transformers.',
      'url': 'ckb/github.com'},
     {'answer': '.',
      'cosine_distance': 0.8993543088436127,
      'date': '22-11-2021',
      'end': 28,
      'qa_score': 1.7178444977616891e-06,
      'start': 27,
      'title': 'Github Library with PyTorch.',
      'url': 'mkb/github.com'}]

    """

    def __init__(self, models: list) -> None:
        self.models = models

    def __repr__(self) -> str:
        repr = "\n".join([model.__repr__() for model in self.models])
        return repr

    def __call__(self, q: str):
        """Compose pipeline

        Parameters
        ----------

            q: Query of the user.

        """
        query = {}
        for model in self.models:
            answer = model(q=q, **query)
            if isinstance(answer, list):
                query.update({"documents": answer})
            elif isinstance(answer, str):
                return answer
        return query["documents"]

    def add(self, documents: list):
        for model in self.models:
            if hasattr(model, "add") and callable(model.add):
                model = model.add(documents=documents)
        return self

    def __add__(self, model):
        """Add a model to the pipeline."""
        self.models.append(model)
        return self
