__all__ = ["Elastic"]

from elasticsearch import Elasticsearch, helpers

from .base import Retriever


class Elastic(Retriever):
    """ElasticSearch retriever.

    Parameters
    ----------
        es: ElasticSearch Python client with selected configuration. Default configuration is used
            if es is set to None.
        index: ElasticSearch index to use.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve
    >>> from elasticsearch import Elasticsearch

    >>> retriever = retrieve.Elastic(on="title", es=Elasticsearch(), index="test")

    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
    ... ]

    >>> retriever = retriever.reset()
    >>> retriever = retriever.add(documents=documents)

    >>> retriever
    Elastic retriever
         on: title
         documents: 3

    >>> print(retriever(q="Transformers", k=2))
    [{'date': '10-11-2021',
      'title': 'Github library with PyTorch and Transformers.',
      'url': 'ckb/github.com'},
     {'date': '22-11-2020',
      'title': 'Github Library with Pytorch and Transformers.',
      'url': 'blp/github.com'}]


    References
    ----------
    1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

    """

    def __init__(self, on: str, es=None, index: str = "nlapi") -> None:
        self.on = on
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
        """ElasticSearch bulk indexing."""
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

    def __call__(self, q: str, k: int = None):
        """ElasticSearch query.

        Parameters
        ----------

            q: User query.
            k: Number of documents to retrieve.
            on: Field to match the query.

        """
        query = {
            "query": {
                "multi_match": {
                    "query": q,
                    "type": "most_fields",
                    "fields": [self.on],
                    "operator": "or",
                }
            },
        }

        if k is not None:
            query["size"] = k

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
