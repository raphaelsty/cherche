import abc
import collections

from ..compose import Intersection, Pipeline, Union

__all__ = ["Query"]


class Query(abc.ABC):
    """Abstract class for models working on a query."""

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
            return other + self
        return Pipeline(models=[other, self])

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
    path_dictionary
        File containing the external vocabulary dictionary.

    """

    def __init__(
        self,
        path_dictionary: str,
    ) -> None:
        super().__init__()
        self.occurrences = collections.Counter()

        with open(path_dictionary, "r") as fp:
            self.path_dictionary = fp.readlines().strip().split("\n")

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> str:
        """Correct spelling errors in a given query."""
        return self
