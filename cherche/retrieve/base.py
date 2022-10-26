import abc
import typing

from ..compose import Intersection, Pipeline, Union, Vote

__all__ = ["Retriever"]


class Retriever(abc.ABC):
    """Retriever base class.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.

    """

    def __init__(
        self, key: str, on: typing.Union[str, list], k: typing.Optional[int]
    ) -> None:
        super().__init__()
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.k = k
        self.documents = None

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n \t key: {self.key}"
        repr += f"\n \t on: {', '.join(self.on)}"
        repr += f"\n \t documents: {self.__len__()}"
        return repr

    @property
    def type(self):
        return "retrieve"

    @abc.abstractclassmethod
    def __call__(self, q: str, **kwargs) -> list:
        return []

    def __len__(self):
        return len(self.documents) if self.documents is not None else 0

    def __add__(self, other) -> Pipeline:
        """Pipeline operator."""
        if isinstance(other, Pipeline):
            return Pipeline(self, other.models)
        elif isinstance(other, list):
            # Documents are part of the pipeline.
            return Pipeline(
                [self, {document[self.key]: document for document in other}]
            )
        return Pipeline([self, other])

    def __or__(self, other) -> Union:
        """Union operator."""
        if isinstance(other, Union):
            return Union([self] + other.models)
        return Union([self, other])

    def __and__(self, other) -> Intersection:
        """Intersection operator."""
        if isinstance(other, Intersection):
            return Intersection([self] + other.models)
        return Intersection([self, other])

    def __mul__(self, other) -> Vote:
        """Voting operator."""
        if isinstance(other, Vote):
            return Vote([self] + other.models)
        return Vote([self, other])
