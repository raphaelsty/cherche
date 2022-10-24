from __future__ import annotations

from .base import Ranker


class Implicit(Ranker):
    """

    """

    def __init__(self, key: str, model, k: int) -> None:
        super().__init__(key, on=[], encoder=None, k=k, similarity=None, store=None)
        self.model = model

    def __call__(self, user: int | str, documents: list, **kwargs) -> list:
        self.model
        return []
