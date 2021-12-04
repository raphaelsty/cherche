__all__ = ["Elastic"]

from elasticsearch import Elasticsearch, helpers


class Elastic:
    """ElasticSearch retriever.

    Parameters
    ----------
        es: ElasticSearch Python client with selected configuration. Default configuration is used
            if es is set to None.
        index: ElasticSearch index to use.

    Examples
    --------
    >>> from anotherrr import retrieve
    >>> from elasticsearch import Elasticsearch


    >>> documents = [
    ...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
    ...     {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
    ...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
    ... ]

    >>> retriever = retrieve.Elastic(
    ...     es = Elasticsearch(),
    ...     index = "test",
    ... )

    >>> model = model.reset_index()

    >>> model = model.add(documents=documents)

    >>> model(q = "Transformers", k=1)


    References
    ----------
        1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

    """

    def __init__(self, on: str, es=None, index: str = "nlapi") -> None:
        self.es = Elasticsearch() if es is None else es
        self.index = index
        self.on = on

        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index)

    def reset(self):
        """Delete the selected index from ElasticSearch."""
        if self.es.indices.exists(index=self.index):
            self.es.delete_by_query(index=self.index, body={"query": {"match_all": {}}})
            self.es.indices.refresh(index=self.index)
        return self

    def add(self, documents: list):
        """ElasticSearch bulk indexing."""
        # Format input documents to add them to ElasticSearch
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

    def __call__(self, q: str, k: int):
        """ElasticSearch query.
        Parameters
        ----------
            q: User query as a string.
            k: Retrieves top k.
            fields: Field to match the query
        """
        query = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": q,
                    "type": "most_fields",
                    "fields": [self.on],
                    "operator": "or",
                }
            },
        }

        documents = self.es.search(
            index=self.index,
            body=query,
        )

        ranked = []

        for document in documents["hits"]["hits"]:
            document["_source"]["score"] = document["_score"]
            ranked.append(document["_source"])

        return ranked
