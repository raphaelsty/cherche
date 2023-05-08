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
    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        k: typing.Optional[int],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.documents = None
        self.k = k
        self.batch_size = batch_size

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n\tkey      : {self.key}"
        repr += f"\n\ton       : {', '.join(self.on)}"
        repr += f"\n\tdocuments: {len(self)}"
        return repr

    @abc.abstractclassmethod
    def __call__(
        self,
        q: typing.Union[typing.List[str], str],
        k: typing.Optional[int],
        batch_size: typing.Optional[int],
        **kwargs,
    ) -> typing.Union[
        typing.List[typing.List[typing.Dict[str, str]]],
        typing.List[typing.Dict[str, str]],
    ]:
        """Retrieve documents from the index."""
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
