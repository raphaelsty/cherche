import abc
import typing

from ..compose import Pipeline


class Generation(abc.ABC):
    """Generation base class.

    Parameters
    ----------
    key
        Field identifier of each document.
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.

    """

    def __init__(
        self,
        on: typing.Union[str, list],
        tokenizer,
        model,
        k: int,
        num_beams: int,
        min_length: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self.on = on
        self.tokenizer = tokenizer
        self.model = model
        self.k = k
        self.num_beams = num_beams
        self.min_length = min_length
        self.max_length = max_length

    def __repr__(self) -> str:
        repr = "Base Generation"
        return repr

    def __add__(self, other) -> Pipeline:
        """Custom operator to make pipeline."""
        if isinstance(other, Pipeline):
            return Pipeline(models=other.models + [self])
        else:
            return Pipeline(models=[other, self])

    def __or__(self) -> None:
        """Or operator is only available on retrievers and rankers."""
        raise NotImplementedError

    def __and__(self) -> None:
        """And operator is only available on retrievers and rankers."""
        raise NotImplementedError
