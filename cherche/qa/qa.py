__all__ = ["QA"]

from operator import itemgetter

from ..compose import Pipeline


class QA:
    """Question Answering model.

    Parameters
    ----------
    model
        Hugging Face question answering model available [here](https://huggingface.co/models?pipeline_tag=question-answering).
    on
        Field to use to answer to the question.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.

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
      'qa_score': 0.2863011956214905,
      'start': 0,
      'title': 'Github Library with Pytorch and Transformers .',
      'url': 'blp/github.com'},
     {'answer': 'Github library',
      'date': '10-11-2021',
      'end': 14,
      'qa_score': 0.2725629210472107,
      'start': 0,
      'title': 'Github library with PyTorch and Transformers .',
      'url': 'ckb/github.com'}]

    """

    def __init__(self, model, on: str, k: int = None) -> None:
        self.model = model
        self.on = on
        self.k = k

    def __repr__(self) -> str:
        repr = "Question Answering"
        repr += f"\n\t model: {self.model.tokenizer.name_or_path}"
        repr += f"\n\t on: {self.on}"
        return repr

    def __call__(self, q: str, documents: list, **kwargs) -> list:
        """Question answering main method.

        Parameters
        ----------
        q
            Input question.
        documents
            List of documents in which the model will retrieve the answer.

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
                        a["qa_score"] = a.pop("score")
                        a.update(**document)
                        top_answers.append(a)
                else:
                    answer["qa_score"] = answer.pop("score")
                    answer.update(**document)
                    top_answers.append(answer)

        answers = sorted(top_answers, key=itemgetter("qa_score"), reverse=True)
        return answers[: self.k] if self.k is not None else answers

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
