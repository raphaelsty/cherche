from .base import Ranker

__all__ = ["DPR"]


class DPR(Ranker):
    def __init__(self) -> None:
        pass

    def __call__(self, documents: list, k: int, on: str) -> list:
        return super().__call__(documents, k, on)
