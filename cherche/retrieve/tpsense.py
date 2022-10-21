import typing

import more_itertools
import tqdm

from .base import Retriever

__all__ = ["Typesense"]


class Typesense(Retriever):
    """Typesense retriever.

    Examples
    --------
    >>> from cherche import retrieve
    >>> import typesense

    >>> documents = [
    ...    {"id": 0, "title": "Paris", "author": "Paris"},
    ...    {"id": 1, "title": "Madrid", "author": "Madrid"},
    ...    {"id": 2, "title": "Montreal", "author": "Montreal"},
    ... ]

    >>> client = typesense.Client({
    ...    'api_key': 'Hu52dwsas2AdxdE',
    ...    'nodes': [{
    ...        'host': 'localhost',
    ...        'port': '8108',
    ...        'protocol': 'http'
    ...    }],
    ...    'connection_timeout_seconds': 2
    ... })

    >>> exist = False
    >>> for collection in client.collections.retrieve():
    ...     if collection["name"] == "documentation":
    ...         exist = True

    >>> if not exist:
    ...     response = client.collections.create({
    ...         "name": "documentation",
    ...         "fields": [
    ...             {"name": "id", "type": "string"},
    ...             {"name": "title", "type": "string"},
    ...             {"name": "author", "type": "string", "optional": True},
    ...         ],
    ...     })

    >>> retriever = retrieve.Typesense(
    ...     key="id",
    ...     on=["title", "author"],
    ...     collection=client.collections['documentation']
    ... )

    >>> retriever.add(documents)
    Typesense retriever
        key: id
        on: title, author
        documents: 3

    >>> retriever("madrid paris")
    [{'author': 'Madrid', 'title': 'Madrid', 'similarity': 1.0, 'id': 1}]

    References
    ----------
    1. [Typesense Github](https://github.com/typesense/typesense)
    2. [Documentation](https://typesense.org/docs/0.23.1/api/)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        collection,
        k: typing.Optional[int] = None,
    ) -> None:
        super().__init__(key, on, k)
        self.collection = collection
        self.integer = False

    def __len__(self):
        return self.collection.retrieve()["num_documents"]

    def add(self, documents, batch_size=128, **kwargs) -> "TypeSense":
        for document in documents:
            if isinstance(document[self.key], int):
                self.integer = True
            break

        for batch in tqdm.tqdm(
            more_itertools.chunked(documents, batch_size),
            position=0,
            desc="TypeSense indexing.",
            total=1 + len(documents) // batch_size,
        ):

            self.collection.documents.import_(
                [{**document, "id": str(document.pop(self.key))} for document in batch],
                {"action": "upsert"},
            )

        return self

    def __call__(self, q: str, query: typing.Optional[dict] = None, **kwargs) -> list:
        """Search for documents.

        Parameters
        ----------
        q
            Query.

        """
        documents = []

        if query is None:
            query = {"q": q, "query_by": self.on}

        for rank, document in enumerate(
            self.collection.documents.search(query)["hits"]
        ):
            document = document["document"]
            document["similarity"] = 1 / (1 + rank)
            document[self.key] = document.pop("id")
            if self.integer:
                document[self.key] = int(document[self.key])

            documents.append(document)
        return documents
