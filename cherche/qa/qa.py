__all__ = ["QA"]

import collections
import typing
from operator import itemgetter

from ..compose import Pipeline
from ..utils import yield_batch


class QA:
    """Question Answering model. QA models needs input documents contents to run.

    Parameters
    ----------
    on
        Fields to use to answer to the question.
    model
        Hugging Face question answering model available [here](https://huggingface.co/models?pipeline_tag=question-answering).

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import retrieve, qa
    >>> from transformers import pipeline

    >>> documents = [
    ...    {"id": 0, "title": "Paris France"},
    ...    {"id": 1, "title": "Madrid Spain"},
    ...    {"id": 2, "title": "Montreal Canada"}
    ... ]

    >>> retriever = retrieve.TfIdf(key="id", on=["title"], documents=documents)

    >>> qa_model = qa.QA(
    ...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
    ...     on = ["title"],
    ...  )

    >>> pipeline = retriever + documents + qa_model

    >>> pipeline
    TfIdf retriever
        key      : id
        on       : title
        documents: 3
    Mapping to documents
    Question Answering
        on: title

    >>> print(pipeline(q="what is the capital of france?"))
    [{'answer': 'Paris',
      'end': 5,
      'id': 0,
      'question': 'what is the capital of france?',
      'score': 0.05615315958857536,
      'similarity': 0.5962847939999439,
      'start': 0,
      'title': 'Paris France'},
     {'answer': 'Montreal',
      'end': 8,
      'id': 2,
      'question': 'what is the capital of france?',
      'score': 0.01080897357314825,
      'similarity': 0.0635641726163728,
      'start': 0,
      'title': 'Montreal Canada'}]

    >>> print(pipeline(["what is the capital of France?", "what is the capital of Canada?"]))
    [[{'answer': 'Paris',
       'end': 5,
       'id': 0,
       'question': 'what is the capital of France?',
       'score': 0.1554129421710968,
       'similarity': 0.5962847939999439,
       'start': 0,
       'title': 'Paris France'},
      {'answer': 'Montreal',
       'end': 8,
       'id': 2,
       'question': 'what is the capital of France?',
       'score': 1.2884755960840266e-05,
       'similarity': 0.0635641726163728,
       'start': 0,
       'title': 'Montreal Canada'}],
     [{'answer': 'Montreal',
       'end': 8,
       'id': 2,
       'question': 'what is the capital of Canada?',
       'score': 0.05316793918609619,
       'similarity': 0.5125692857821978,
       'start': 0,
       'title': 'Montreal Canada'},
      {'answer': 'Paris France',
       'end': 12,
       'id': 0,
       'question': 'what is the capital of Canada?',
       'score': 4.7594025431862974e-07,
       'similarity': 0.035355339059327376,
       'start': 0,
       'title': 'Paris France'}]]

    """

    def __init__(
        self,
        on: typing.Union[str, list],
        model,
        batch_size: int = 32,
    ) -> None:
        self.on = on if isinstance(on, list) else [on]
        self.model = model
        self.batch_size = batch_size

    def __repr__(self) -> str:
        repr = "Question Answering"
        repr += f"\n\ton: {', '.join(self.on)}"
        return repr

    def get_question_context(
        self,
        q: typing.List[str],
        documents: typing.List[typing.List[typing.Dict[str, str]]],
    ):
        question_context = []
        for query, documents_query in zip(q, documents):
            for document in documents_query:
                question_context.append(
                    (query, " ".join([document.get(field, " ") for field in self.on]))
                )
        return question_context

    def __call__(
        self,
        q: typing.Union[str, typing.List[str]],
        documents: typing.Union[
            typing.List[typing.List[typing.Dict[str, str]]],
            typing.List[typing.Dict[str, str]],
        ],
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Question answering main method.

        Parameters
        ----------
        q
            Input question.
        documents
            List of documents within which the model will retrieve the answer.

        """
        if isinstance(q, str) and not documents:
            return []

        if isinstance(q, list) and not documents:
            return [[]]

        if isinstance(q, str):
            questions = [q]
            documents = [documents]
        else:
            questions = q

        question_context = self.get_question_context(questions, documents)

        answers = []
        for batch in yield_batch(
            question_context,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            desc="Question answering",
        ):
            answers.extend(
                self.model(
                    {
                        "question": [question for question, _ in batch],
                        "context": [context for _, context in batch],
                    },
                )
            )

        answers = collections.deque(answers)
        ranked = [
            sorted(
                [
                    {**document, **answers.popleft(), **{"question": question}}
                    for document in documents_query
                ],
                key=itemgetter("score"),
                reverse=True,
            )
            for question, documents_query in zip(questions, documents)
        ]

        return ranked[0] if isinstance(q, str) else ranked

    def __add__(self, other) -> Pipeline:
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return Pipeline(models=other.models + [self])
        else:
            return Pipeline(models=[other, self])

    def __or__(self) -> None:
        """Or operator is only available for retrievers and rankers."""
        raise NotImplementedError

    def __and__(self) -> None:
        """And operator is only available for retrievers and rankers."""
        raise NotImplementedError

    def __mul__(self) -> None:
        """Mul operator is only available for retrievers and rankers."""
        raise NotImplementedError
