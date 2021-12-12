__all__ = ["Summary"]

from ..compose import Pipeline


class Summary:
    """Summarization model. Returns a single summary for inputs documents.

    Parameters
    ----------

        model: Summarization pipeline from HuggingFace.
        on: Fild of documents to use to create the summary.
        min_lenght: Minimum number of token of the summary.
        max_lenght: Maximum number of token of the summary.


    Examples
    --------

    >>> from transformers import pipeline
    >>> from cherche import summary

    >>> model = summary.Summary(
    ...    model = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6", framework="pt"),
    ...    on = "title",
    ... )

    >>> model
    Summarization model
         on: title
         min length: 5
         max length: 30

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "CKB is a Github library with PyTorch and Transformers.", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "MKB Github Library with PyTorch  dedicated to KB.", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "BLP is a Github Library with Pytorch and Transformers dedicated to KB.", "date": "22-11-2020"},
    ... ]

    >>> print(model(documents=documents))
    CKB is a Github library with Pytorch and Transformers dedicated to KB. MKB Github Library with PyTorch  dedicated to

    """

    def __init__(
        self,
        model,
        on: str,
        min_length: int = 5,
        max_length: int = 30,
    ) -> None:
        self.model = model
        self.on = on
        self.min_length = min_length
        self.max_length = max_length

    def __repr__(self) -> str:
        repr = "Summarization model"
        repr += f"\n\t on: {self.on}"
        repr += f"\n\t min length: {self.min_length}"
        repr += f"\n\t max length: {self.max_length}"
        return repr

    def __call__(self, documents: list, **kwargs) -> list:
        """Summarize input text.

        Parameters
        ----------

            documents: List of documents to summarize.

        """
        if not documents:
            return []

        return self.model(
            " ".join([document[self.on] for document in documents]),
            min_length=self.min_length,
            max_length=self.max_length,
            return_text=True,
            clean_up_tokenization_spaces=True,
            return_tensors=False,
        )[0]["summary_text"].strip()

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return other + self
        else:
            return Pipeline(models=[other, self])

    def __or__(self):
        """Or operator is only available on retrievers and rankers."""
        raise NotImplementedError

    def __and__(self):
        """And operator is only available on retrievers and rankers."""
        raise NotImplementedError
