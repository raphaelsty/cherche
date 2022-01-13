import abc

__all__ = ["Compose"]


class Compose(abc.ABC):
    """Base class for Pipeline."""

    def __init__(self, models: list) -> None:
        self.models = models
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> list:
        return []

    def add(self, documents: list) -> "Compose":
        for model in self.models:
            if hasattr(model, "add") and callable(model.add):
                model = model.add(documents=documents)
        return self

    def reset(self) -> "Compose":
        for model in self.models:
            if hasattr(model, "reset") and callable(model.reset):
                model = model.reset()
        return self

    def __repr__(self) -> str:
        repr = "\n".join(
            [
                model.__repr__() if not isinstance(model, dict) else "Mapping to documents"
                for model in self.models
            ]
        )
        return repr
