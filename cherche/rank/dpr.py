from .base import Ranker

__all__ = ["DPR"]


class DPR(Ranker):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, documents: list[dict], k: int, on: str) -> list:
        return super().__call__(documents, k, on)
