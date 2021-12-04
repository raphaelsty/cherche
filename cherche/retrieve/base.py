import abc

__all__ = ["Retriever"]


class Retriever(abc.ABC):
    """Retriever base class."""

    def __init__(self, on: str) -> None:
        super().__init__()
        self.on = on
        self.documents = []

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n \t on: {self.on}"
        repr += f"\n \t documents: {self.__len__()}"
        return repr

    @abc.abstractclassmethod
    def __call__(self, q: list, k: int = None) -> list:
        pass

    @abc.abstractclassmethod
    def add(self, documents: list):
        return self

    def __len__(self):
        return len(self.documents)
