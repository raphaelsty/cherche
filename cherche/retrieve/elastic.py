__all__ = ["Elastic"]

from elasticsearch import Elasticsearch, helpers

from .base import Retriever


class Elastic(Retriever):
    """ElasticSearch retriever based on the [Python client of Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v7.15.1/).

    Parameters
    ----------
    on
        Field to use to match the query to the documents.
    k
        Number of documents to retrieve. Default is None, i.e all documents that match the query
        will be retrieved.
    es
        ElasticSearch Python client. The default configuration is used if set to None.
    index
        Elasticsearch index to use to index documents. Elastic will create the index if it does not
        exist.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from elasticsearch import Elasticsearch

    >>> es = Elasticsearch()

    >>> if es.ping():
    ...
    ...     retriever = retrieve.Elastic(on="article", k=2, es=es, index="test")
    ...
    ...     documents = [
    ...         {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...         {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...         {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ...     ]
    ...
    ...     retriever = retriever.reset()
    ...     retriever = retriever.add(documents=documents)
    ...     candidates = retriever(q="paris")

    References
    ----------
    1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

    """

    def __init__(self, on: str, k: int = None, es=None, index: str = "cherche") -> None:
        self.on = on
        self.k = k
        self.es = Elasticsearch() if es is None else es
        self.index = index

        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index)

    def __len__(self):
        return len(
            self.es.search(index=self.index, body={"query": {"match_all": {}}},)[
                "hits"
            ]["hits"]
        )

    def add(self, documents: list):
        """ElasticSearch bulk indexing.

        Parameters
        ----------
        documents
            List of documents to upload to Elasticsearch.

        """
        documents = [
            {
                "_index": self.index,
                "_source": doc,
            }
            for doc in documents
        ]
        helpers.bulk(self.es, documents)
        self.es.indices.refresh(index=self.index)
        return self

    def __call__(self, q: str):
        """ElasticSearch query.

        Parameters
        ----------

            q: User query.
            on: Field to match the query.

        """
        query = {"query": {"match": {self.on: q}}}

        if self.k is not None:
            query["size"] = self.k

        documents = self.es.search(
            index=self.index,
            body=query,
        )

        return [document["_source"] for document in documents["hits"]["hits"]]

    def reset(self):
        """Delete the selected index from ElasticSearch."""
        if self.es.indices.exists(index=self.index):
            self.es.delete_by_query(index=self.index, body={"query": {"match_all": {}}})
            self.es.indices.refresh(index=self.index)
        return self
