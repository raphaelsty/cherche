import abc

from cherche.pipeline.pipeline import Pipeline

__all__ = ["Retriever"]


class Retriever(abc.ABC):
    """Retriever base class."""

    def __init__(self, on: str, k: int) -> None:
        super().__init__()
        self.on = on
        self.k = k
        self.documents = []

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n \t on: {self.on}"
        repr += f"\n \t documents: {self.__len__()}"
        return repr

    @abc.abstractclassmethod
    def __call__(self, q: str) -> list:
        pass

    @abc.abstractclassmethod
    def add(self, documents: list):
        return self

    def __len__(self):
        return len(self.documents)

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return other + self
        else:
            return Pipeline(models=[self, other])
