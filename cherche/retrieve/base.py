import abc

from ..compose import Compose

__all__ = ["Retriever"]


class Retriever(abc.ABC):
    """Retriever base class."""

    def __init__(self, on: str, k: int) -> None:
        super().__init__()
        self.on = on
        self.k = k
        self.documents = []

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} retriever"
        repr += f"\n \t on: {self.on}"
        repr += f"\n \t documents: {self.__len__()}"
        return repr

    @abc.abstractclassmethod
    def __call__(self, q: str) -> list:
        pass

    @abc.abstractclassmethod
    def add(self, documents: list):
        return self

    def __len__(self):
        return len(self.documents)

    def __add__(self, other):
        """Custom operator to make pipeline."""
        if isinstance(other, Compose):
            return other + self
        else:
            return Compose(models=[self, other])


class _BM25(Retriever):
    """Base class for BM25, BM25L and BM25Okapi Retriever.

    Parameters
    ----------

        on: Field that BM25 will use to search relevant documents.
        bm25: Model from https://github.com/dorianbrown/rank_bm25.
        tokenizer: Default tokenizer consist by splitting on space. This tokenizer should have a
            tokenizer.__call__ method that returns the list of tokens from an input sentence.
        k: Number of documents to retrieve.

    """

    def __init__(self, on: str, bm25, tokenizer=None, k: int = None) -> None:
        super().__init__(on=on, k=k)
        self.bm25 = bm25
        self.tokenizer = tokenizer

    def add(self, documents: list):
        """Add documents."""
        self.documents += documents
        self.bm25 = self.bm25(
            [
                doc[self.on].split(" ") if self.tokenizer is None else self.tokenizer(doc[self.on])
                for doc in self.documents
            ]
        )
        return self

    def __call__(self, q: str) -> list:
        """Retrieve the right document using BM25."""
        q = q.split(" ") if self.tokenizer is None else self.tokenizer(q)
        similarities = abs(self.bm25.get_scores(q))
        documents = [self.documents[index] for index in (-similarities).argsort()]
        return documents[: self.k] if self.k is not None else documents
