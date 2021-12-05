__all__ = ["QA"]

from operator import itemgetter

from ..compose import Compose


class QA:
    """Question Answering model.

    Parameters
    ----------

        model: HuggingFace QA model.
        on: Field to use to answer the questions.

    Examples
    --------


    >>> from pprint import pprint as print
    >>> from transformers import pipeline
    >>> from cherche import qa

    >>> model = qa.QA(
    ...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
    ...     on = "title",
    ...  )

    >>> model
    Question Answering
         model: deepset/roberta-base-squad2
         on: title

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
    ... ]

    >>> print(model(q="What is used with Transformers?", documents=documents, k=2))
    [{'answer': 'Github Library',
      'date': '22-11-2020',
      'end': 14,
      'score': 0.2863011956214905,
      'start': 0,
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'},
     {'answer': 'Github library',
      'date': '10-11-2021',
      'end': 14,
      'score': 0.2725629210472107,
      'start': 0,
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'}]

    """

    def __init__(self, model, on: str) -> None:
        self.model = model
        self.on = on

    def __repr__(self) -> str:
        repr = "Question Answering"
        repr += f"\n\t model: {self.model.tokenizer.name_or_path}"
        repr += f"\n\t on: {self.on}"
        return repr

    def __call__(self, q: str, documents: list, k: int = None, **kwargs) -> list:
        """Question answering main method.

        Parameters
        ----------

            q: Question.
            documents: List of documents in which the model will retrieve the answer.
            k: Number of candidates answers to retrieve.

        """
        if not documents:
            return []

        answers = self.model(
            {
                "question": [q for _ in documents],
                "context": [document[self.on] for document in documents],
            },
        )

        top_answers = []
        for answer, document in zip(answers, documents):
            if isinstance(document, dict):
                if isinstance(answer, list):
                    for a in answer:
                        a.update(**document)
                        top_answers.append(a)
                else:
                    answer.update(**document)
                    top_answers.append(answer)

        answers = sorted(top_answers, key=itemgetter("score"), reverse=True)
        return answers[:k] if k is not None else answers

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Compose):
            return other + self
        else:
            return Compose(models=[other, self])
