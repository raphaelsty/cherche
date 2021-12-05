__all__ = ["ZeroShot"]


class ZeroShot:
    """ZeroShot

    Parameters
    ----------

        encoder: HuggingFace pipeline for zero shot classification.
        on: Field to use for the zero shot classification.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import rank
    >>> from transformers import pipeline

    >>> ranker = rank.ZeroShot(
    ...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
    ...     on = "title",
    ... )

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    >>> print(ranker(q="Transformers", documents=documents, k=2))
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

    def __init__(
        self,
        encoder,
        on: str,
    ):
        self.encoder = encoder
        self.on = on

    def __call__(self, q: str, documents: list, k: int = None, multi_label: bool = True):
        """Rank inputs documents based on query.

        Parameters
        ----------

            q: Query of the user.
            documents: List of documents to rank.
            on: Field of documents to use for ranking.

        """
        scores = self.encoder(
            q,
            [document[self.on] for document in documents],
            multi_label=multi_label,
        )

        ranked = []
        for rank, (label, score) in enumerate(zip(scores["labels"], scores["scores"])):
            for index, document in enumerate(documents):
                if document[self.on] == label:
                    ranked.append(document)
                    document.update({"_zero_shot_score": score})
                    documents.pop(index)
                    break

            if k is not None:
                if rank + 1 >= k:
                    break

        return ranked
