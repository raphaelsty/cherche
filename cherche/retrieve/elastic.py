__all__ = ["Elastic"]

import typing

import more_itertools
import tqdm
from elasticsearch import Elasticsearch, helpers

from .base import Retriever


class Elastic(Retriever):
    """ElasticSearch retriever based on the [Python client of Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v7.15.1/).

    Parameters
    ----------
    on
        Fields to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is `None`, i.e all documents that match the query
         will be retrieved.
    es
        ElasticSearch Python client. The default configuration is used if set to None.
    index
        Elasticsearch index to use to index documents. Elastic will create the index if it does not
        exist.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from elasticsearch import Elasticsearch
    >>> from cherche import retrieve, rank
    >>> from sentence_transformers import SentenceTransformer

    >>> es = Elasticsearch(hosts="http://localhost:9200")

    >>> if es.ping():
    ...    retriever = retrieve.Elastic(key="id", on=["title", "author"], k=2, es=es, index="test")
    ...
    ...    documents = [
    ...         {"id": 0, "title": "Paris", "author": "Wiki"},
    ...         {"id": 1, "title": "Eiffel tower", "author": "Wiki"},
    ...         {"id": 2, "title": "Montreal", "author": "Wiki"},
    ...    ]
    ...
    ...    retriever = retriever.add(documents=documents)
    ...    candidates = retriever(q="paris")

    References
    ----------
    1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

    """

    def __init__(
        self,
        key: str,
        on: typing.Union[str, list],
        k: int = None,
        es: Elasticsearch = None,
        index: str = "cherche",
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.k = k
        self.es = Elasticsearch() if es is None else es
        self.index = index

        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index)

    def __len__(self) -> int:
        return int(self.es.cat.count(self.index, params={"format": "json"})[0]["count"])

    def add(self, documents: list, batch_size=128, **kwargs) -> "Elastic":
        """ElasticSearch bulk indexing.

        Parameters
        ----------
        documents
            List of documents to upload to Elasticsearch.

        """
        documents = [
            {
                "_id": doc[self.key],
                "_index": self.index,
                "_source": doc,
            }
            for doc in documents
        ]

        for batch in tqdm.tqdm(
            more_itertools.chunked(documents, batch_size),
            position=0,
            desc="Elasticsearch indexing.",
            total=1 + len(documents) // batch_size,
        ):

            helpers.bulk(self.es, batch)
            self.es.indices.refresh(index=self.index)

        return self

    def __call__(self, q: str, query: str = None, **kwargs) -> list:
        """ElasticSearch query.

        Parameters
        ----------

            q: User query.
            query: Custom ElasticSearch query.

        """
        if query is None:
            query = {
                "query": {
                    "multi_match": {
                        "query": q,
                        "type": "most_fields",
                        "fields": self.on,
                        "operator": "or",
                    }
                },
            }

        if self.k is not None:
            query["size"] = self.k

        documents = self.es.search(
            index=self.index,
            body=query,
        )

        ranked = []
        for document in documents["hits"]["hits"]:
            document = {**document["_source"], "similarity": float(document["_score"])}
            ranked.append(document)

        return ranked

    def reset(self) -> "Elastic":
        """Delete the selected index from ElasticSearch."""
        if self.es.indices.exists(index=self.index):
            self.es.delete_by_query(index=self.index, body={"query": {"match_all": {}}})
            self.es.indices.refresh(index=self.index)
        return self
