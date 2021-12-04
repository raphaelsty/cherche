import abc

__all__ = ["Ranker"]


class Ranker(abc.ABC):
    """Abstract class for ranking models."""

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        pass

    @abc.abstractmethod
    def __call__(self, documents: list[dict], k: int, on: str) -> list:
        pass
