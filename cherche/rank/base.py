import abc

__all__ = ["Ranker"]


class Ranker(abc.ABC):
    """Abstract class for ranking models."""

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} ranker"
        repr += f"\n \t on: {self.on}"
        return repr

    @abc.abstractmethod
    def __call__(self, documents: list, on: str, k: int = None) -> list:
        pass
