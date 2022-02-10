import abc
import collections
import re
import typing

from ..compose import Intersection, Pipeline, Union

__all__ = ["Query"]


class Query(abc.ABC):
    """Abstract class for models working on a query."""

    def __init__(self, on: typing.Union[str, list]):
        self.on = on if isinstance(on, list) else [on]

    @property
    def type(self):
        return "query"

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} query"
        return repr

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> str:
        return self

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, Pipeline):
            return self + other
        return Pipeline(models=[self] + other.models)

    def __or__(self, other) -> Union:
        """Union operator."""
        raise NotImplementedError("Union not working with a Query model")

    def __and__(self, other) -> Intersection:
        """Intersection operator."""
        raise NotImplementedError("Intersection not working with a Query model")


class _SpellingCorrector(Query):
    """Base class for the Norvig spelling corrector model.

    Parameters
    ----------
    on
        Fields to use for fitting the spelling corrector on.

    """

    def __init__(
        self,
        on: typing.Union[str, list],
    ) -> None:
        super().__init__(on=on)
        self.occurrences = collections.Counter()

    def add(
        self, documents: typing.Union[typing.List[typing.Dict], str]
    ) -> "_SpellingCorrector":
        if isinstance(documents, str):
            self._update_from_str(documents)
        elif isinstance(documents, list) and len(documents) > 0:
            if isinstance(documents[0], dict):
                text = " ".join(
                    [
                        " ".join([document.get(field, "") for field in self.on])
                        for document in documents
                    ]
                )
                self._update_from_str(text)
        else:
            raise ValueError(
                f"Unsupported document format for updating spelling dictionary : {type(documents)}"
            )
        return self

    def reset(self) -> "_SpellingCorrector":
        """Wipe dictionary."""
        self.occurrences = collections.Counter()
        return self

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> str:
        """Correct spelling errors in a given query."""
        return self

    def _update_from_str(self, words: str):
        """Update dictionary from all words presents in a string."""
        words = words.split(" ")
        self._update_from_list(words=words)

    def _update_from_list(self, words: typing.List[str]):
        """Update dictionary from all words presents in a list of strings."""
        self.occurrences.update(words)

    def _update_from_file(self, path_file: str):
        """Update dictionary from all words fetched from a raw text file."""
        with open(path_file, "r") as fp:
            words = re.findall(r"\w+", fp.read().lower())
            self._update_from_list(words=words)
