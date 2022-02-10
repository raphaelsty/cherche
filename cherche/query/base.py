import abc
import typing

from ..compose import Intersection, Pipeline, Union

__all__ = ["Query"]


class Query(abc.ABC):
    """Abstract class for models working on a query."""

    def __init__(self, on: typing.Union[str, list]):
        self.on = on if isinstance(on, list) else [on]

    @property
    def type(self) -> str:
        return "query"

    def __repr__(self) -> str:
        repr = f"Query {self.__class__.__name__}"
        return repr

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> str:
        return self

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, Pipeline):
            return Pipeline(models=[self] + other.models)
        return Pipeline(models=[self, other])

    def __or__(self, other) -> Union:
        """Union operator."""
        raise NotImplementedError("Union not working with a Query model")

    def __and__(self, other) -> Intersection:
        """Intersection operator."""
        raise NotImplementedError("Intersection not working with a Query model")
