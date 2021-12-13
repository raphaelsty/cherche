__all__ = ["Summary"]

from ..compose import Pipeline


class Summary:
    """Summarization model. Returns a single summary for an input list of documents.

    Parameters
    ----------
    model
        Hugging Face summarization model available [here](https://huggingface.co/models?pipeline_tag=summarization).
    on
        Field to use to answer to the question.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.
    min_lenght
        Minimum number of token of the summary.
    max_lenght
        Maximum number of token of the summary.

    Examples
    --------

    >>> from transformers import pipeline
    >>> from cherche import summary

    >>> model = summary.Summary(
    ...    model = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6", framework="pt"),
    ...    on = "article",
    ... )

    >>> model
    Summarization model
         on: article
         min length: 5
         max length: 30

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> print(model(documents=documents))
    This town is the capital of France Eiffel tower is based in Paris Montreal is in Canada.

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

    def __call__(self, documents: list, **kwargs) -> str:
        """Summarize input text.

        Parameters
        ----------
        documents
            List of documents to summarize.

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
