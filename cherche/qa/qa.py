__all__ = ["QA"]

import typing
from operator import itemgetter

from ..compose import Pipeline


class QA:
    """Question Answering model.

    Parameters
    ----------
    on
        Fields to use to answer to the question.
    model
        Hugging Face question answering model available [here](https://huggingface.co/models?pipeline_tag=question-answering).
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
        will be retrieved.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from transformers import pipeline
    >>> from cherche import qa

    >>> documents = [
    ...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> model = qa.QA(
    ...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
    ...     on = ["title", "article"],
    ...     k = 2,
    ...  )

    >>> model
    Question Answering
         model: deepset/roberta-base-squad2
         on: title, article

    >>> print(model(q="Where is the Eiffel tower?", documents=documents))
    [{'answer': 'Paris',
      'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'end': 43,
      'qa_score': 0.9743021130561829,
      'start': 38,
      'title': 'Eiffel tower'},
     {'answer': 'Paris',
      'article': 'This town is the capital of France',
      'author': 'Wiki',
      'end': 5,
      'qa_score': 0.0003580129996407777,
      'start': 0,
      'title': 'Paris'}]

    """

    def __init__(self, on: typing.Union[str, list], model, k: int = None) -> None:
        self.on = on if isinstance(on, list) else [on]
        self.model = model
        self.k = k

    @property
    def type(self):
        return "qa"

    def __repr__(self) -> str:
        repr = "Question Answering"
        repr += f"\n\t model: {self.model.tokenizer.name_or_path}"
        repr += f"\n\t on: {', '.join(self.on)}"
        return repr

    def __call__(self, q: str, documents: list, **kwargs) -> list:
        """Question answering main method.

        Parameters
        ----------
        q
            Input question.
        documents
            List of documents within which the model will retrieve the answer.

        """
        if not documents:
            return []

        answers = self.model(
            {
                "question": [q for _ in documents],
                "context": [
                    " ".join([document.get(field, "") for field in self.on])
                    for document in documents
                ],
            },
        )

        top_answers = []
        for answer, document in zip(answers, documents):
            if isinstance(document, dict):
                if isinstance(answer, list):
                    for a in answer:
                        a["qa_score"] = a.pop("score")
                        a.update(**document)
                        top_answers.append(a)
                else:
                    answer["qa_score"] = answer.pop("score")
                    answer.update(**document)
                    top_answers.append(answer)

        answers = sorted(top_answers, key=itemgetter("qa_score"), reverse=True)
        return answers[: self.k] if self.k is not None else answers

    def __add__(self, other) -> Pipeline:
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return Pipeline(models=other.models + [self])
        else:
            return Pipeline(models=[other, self])

    def __or__(self) -> None:
        """Or operator is only available on retrievers and rankers."""
        raise NotImplementedError

    def __and__(self) -> None:
        """And operator is only available on retrievers and rankers."""
        raise NotImplementedError
