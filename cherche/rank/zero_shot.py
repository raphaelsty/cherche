__all__ = ["ZeroShot"]

from ..compose import Compose


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

    >>> ranker = rank.ZeroShot(
    ...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
    ...     on = "title",
    ...     k = 2,
    ... )

    >>> ranker
    Zero Shot Classifier
         model: typeform/distilbert-base-uncased-mnli
         on: title
         k: 2
         multi class: True

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    >>> print(ranker(q="Transformers", documents=documents))
    [{'_zero_shot_score': 0.3513341546058655,
      'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'},
     {'_zero_shot_score': 0.3513341546058655,
      'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'}]

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
        for rank, (label, score) in enumerate(zip(scores["labels"], scores["scores"])):
            for index, document in enumerate(documents):
                if document[self.on] == label:
                    ranked.append(document)
                    document.update({"_zero_shot_score": score})
                    documents.pop(index)
                    break

            if self.k is not None:
                if rank + 1 >= self.k:
                    break

        return ranked

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Compose):
            return other + self
        else:
            return Compose(models=[other, self])
