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

    def add(self, documents: list, **kwargs) -> "Compose":
        """Add new documents."""
        history = {}
        for model in self.models:
            if hasattr(model, "add") and callable(model.add):

                # Avoid indexing twice the same model, index or store.
                if id(model) in history:
                    continue
                if hasattr(model, "index"):
                    if id(model.index) in history:
                        continue
                if hasattr(model, "store"):
                    if id(model.store) in history:
                        continue

                model = model.add(documents=documents, **kwargs)

                history[id(model)] = True
                if hasattr(model, "index"):
                    history[model.index] = True

                if hasattr(model, "store"):
                    history[model.store] = True

        return self

    def reset(self) -> "Compose":
        for model in self.models:
            if hasattr(model, "reset") and callable(model.reset):
                model = model.reset()
        return self

    def __repr__(self) -> str:
        repr = "\n".join(
            [
                model.__repr__()
                if not isinstance(model, dict)
                else "Mapping to documents"
                for model in self.models
            ]
        )
        return repr
