__all__ = ["ZeroShot"]

from ..compose import Intersection, Pipeline, Union


class ZeroShot:
    """ZeroShot classifier for ranking.

    Parameters
    ----------

        encoder: HuggingFace pipeline for zero shot classification.
        on: Field to use for the zero shot classification.
        k: Number of documents to keep.
        multi_class: If more than one candidate label can be correct, pass multi_class=True to
            calculate each class independently.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from transformers import pipeline
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> ranker = rank.ZeroShot(
    ...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
    ...     on = "article",
    ...     k = 2,
    ... )

    >>> ranker
    Zero Shot Classifier
         model: typeform/distilbert-base-uncased-mnli
         on: article
         k: 2
         multi class: True

    >>> print(ranker(q="Paris", documents=documents))
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'similarity': 0.4519128203392029,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'similarity': 0.33974456787109375,
      'title': 'Eiffel tower'}]

    References
    ----------
    1. [New pipeline for zero-shot text classification](https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681)
    2. [NLI models](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads)

    """

    def __init__(self, encoder, on: str, k: int = None, multi_class: bool = True):
        self.encoder = encoder
        self.on = on
        self.k = k
        self.multi_class = multi_class

    def __repr__(self) -> str:
        repr = "Zero Shot Classifier"
        repr += f"\n\t model: {self.encoder.tokenizer.name_or_path}"
        repr += f"\n\t on: {self.on}"
        repr += f"\n\t k: {self.k}"
        repr += f"\n\t multi class: {self.multi_class}"
        return repr

    def __call__(self, q: str, documents: list, **kwargs):
        """Rank inputs documents based on query.

        Parameters
        ----------

            q: Query of the user.
            documents: List of documents to rank.

        """
        if not documents:
            return []

        scores = self.encoder(
            q,
            [document[self.on] for document in documents],
            multi_class=self.multi_class,
        )

        ranked = []
        for label, score in zip(scores["labels"], scores["scores"]):
            for document in documents:
                if document[self.on] == label:
                    ranked.append(document)
                    document.update({"similarity": score})
        return ranked[: self.k] if self.k is not None else ranked

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return other + self
        else:
            return Pipeline(models=[other, self])

    def __or__(self, other):
        """Custom operator for union."""
        if isinstance(other, Union):
            return Union(models=[self] + [other.models])
        else:
            return Union(models=[self, other])

    def __and__(self, other):
        """Custom operator for intersection."""
        if isinstance(other, Union):
            return Intersection(models=[self] + [other.models])
        else:
            return Intersection(models=[self, other])

    def add(self, documents):
        """Zero shot do not pre-compute embeddings."""
        return self
