import copy
import typing

from ..compose import Pipeline

__all__ = ["Translate", "TranslateQuery"]


class BaseTranslate:
    def __init__(self, translation) -> None:
        self.translation = translation

    @property
    def type(self):
        return "translate"

    def __or__(self) -> None:
        """Or operator is only available on retrievers and rankers."""
        raise NotImplementedError

    def __and__(self) -> None:
        """And operator is only available on retrievers and rankers."""
        raise NotImplementedError


class Translate(BaseTranslate):
    """Translation module using Hugging Face pre-trained models.

    Parameters
    ----------
    on
        Field to use to retrieve documents.
    translation
        Hugging Face translation pipeline.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import translate, retrieve
    >>> from transformers import pipeline

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> translation = translate.Translate(
    ...     on = ["title", "article"],
    ...     translation = pipeline("translation_en_to_fr", model = "t5-small"),
    ... )

    >>> translation
    Translation
         on: ['title', 'article']

    >>> print(translation(documents=documents))
    [{'article': 'Cette ville est la capitale française',
      'author': 'Wiki',
      'id': 0,
      'title': 'Paris'},
     {'article': 'La tour Eiffel est basée à Paris.',
      'author': 'Wiki',
      'id': 1,
      'title': 'Tour Eiffel'},
     {'article': 'Montréal est au Canada.',
      'author': 'Wiki',
      'id': 2,
      'title': 'Montréal'}]

     >>> search = (retrieve.TfIdf(key = "id", on = "title", k=1, documents=documents) +
     ...    documents + translation)

     >>> print(search("Paris"))
     [{'article': 'Cette ville est la capitale française',
      'author': 'Wiki',
      'id': 0,
      'title': 'Paris'}]

     >>> print(documents)
     [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'title': 'Eiffel tower'},
     {'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'title': 'Montreal'}]

    References
    ----------
    1. [Hugging Face](https://huggingface.co/models?pipeline_tag=translation)

    """

    def __init__(self, on: typing.Union[list, str], translation) -> None:
        super().__init__(translation=translation)
        self.on = on if isinstance(on, list) else [on]

    def __repr__(self) -> str:
        repr = "Translation"
        repr += f"\n\t on: {self.on}"
        return repr

    def __call__(
        self,
        documents: list,
        **kwargs,
    ) -> typing.Union[str, list]:
        """Translate documents. Translate all documents per batch and finally update documents.

        Parameters
        ----------
        documents
            List of documents to translate.

        """
        translated_documents = self.translation(
            [
                item
                for sublist in [
                    [document.get(field, "") for field in self.on] for document in documents
                ]
                for item in sublist
            ]
        )

        documents = copy.deepcopy(documents)
        for document in documents:
            for index, (content, field) in enumerate(zip(translated_documents, self.on)):
                if index >= len(self.on):
                    break
                document[field] = content["translation_text"]
            for field in self.on:
                del translated_documents[0]

        return documents

    def __add__(self, other) -> Pipeline:
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return other + self
        else:
            return Pipeline(models=[other, self])


class TranslateQuery(BaseTranslate):
    """Translation module using Hugging Face pre-trained models.

    Parameters
    ----------
    translation
        Hugging Face translation pipeline.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import translate, retrieve
    >>> from transformers import pipeline

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ... ]

    >>> query_translation = translate.TranslateQuery(
    ...     translation = pipeline("translation_fr_to_en", model = "Helsinki-NLP/opus-mt-fr-en"),
    ... )

    >>> search = query_translation + retrieve.TfIdf(key = "id", on = "article", k=1, documents=documents)

    >>> query_translation("tour eiffel")
    'eiffel tower'

    >>> print(search("eiffel tower"))
    [{'id': 1}]

    References
    ----------
    1. [Hugging Face](https://huggingface.co/models?pipeline_tag=translation)

    """

    def __init__(self, translation) -> None:
        super().__init__(translation=translation)

    def __repr__(self) -> str:
        return "Query translation"

    def __call__(self, q: str, **kwargs) -> str:
        return self.translation(q)[0]["translation_text"]

    def __add__(self, other) -> Pipeline:
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return other + self
        else:
            return Pipeline(models=[self, other])
