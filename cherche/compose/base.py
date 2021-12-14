import abc

__all__ = ["Compose"]


class Compose(abc.ABC):
    """Base class for Pipeline."""

    def __init__(self, models: list) -> None:
        self.models = models

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> list:
        return []

    def add(self, documents: list) -> "Compose":
        for model in self.models:
            if hasattr(model, "add") and callable(model.add):
                model = model.add(documents=documents)
        return self

    def __repr__(self) -> str:
        repr = "\n".join([model.__repr__() for model in self.models])
        return repr

    def __add__(self, other) -> "Compose":
        """Pipeline operator."""
        self.models.append(other)
        return self
