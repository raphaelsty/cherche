__all__ = ["Elastic"]

import typing

import numpy as np
from elasticsearch import Elasticsearch, helpers

from ..rank import Ranker
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
    >>> from cherche import retrieve
    >>> from elasticsearch import Elasticsearch

    >>> es = Elasticsearch()

    >>> if es.ping():
    ...
    ...     retriever = retrieve.Elastic(key="id", on=["title", "article"], k=2, es=es, index="test")
    ...
    ...     documents = [
    ...         {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
    ...         {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
    ...         {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
    ...     ]
    ...
    ...     retriever = retriever.add(documents=documents)
    ...     candidates = retriever(q="paris")

    >>> print(candidates)
    [{'article': 'This town is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'similarity': 1.2017119,
      'title': 'Paris'},
     {'article': 'Eiffel tower is based in Paris',
      'author': 'Wiki',
      'id': 1,
      'similarity': 1.0534589,
      'title': 'Eiffel tower'}]

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

    def add_embeddings(
        self, documents: list, ranker: Ranker = None, embeddings: list = None
    ) -> "Elastic":
        """Store documents and embeddings inside Elasticsearch using bulk indexing. Embeddings
        parameter has the priority over ranker. If embeddings are provided, ElasticSearch will
        index documents with their embeddings. If embeddings are not provided, the Ranker will
        be called to compute embeddings. This method is useful if you have to deal with large
        corpora.

        Parameters
        ----------
        documents
            List of documents to upload to Elasticsearch.
        ranker
              Elastic can store embeddings of the ranker to limit ram usage. If provided, when
              elastic retrieves documents, it will retrieve embeddings also. If not provided, it
              index embeddings.
        embeddings
            Elastic can store embeddings of the ranker to limit ram usage.

        Examples
        --------

        >>> from cherche import retrieve, rank
        >>> from sentence_transformers import SentenceTransformer
        >>> from elasticsearch import Elasticsearch

        >>> documents = [
        ...     {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
        ...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
        ...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
        ... ]

        >>> es = Elasticsearch()

        >>> if es.ping():
        ...
        ...    ranker = rank.Encoder(
        ...         key = "id",
        ...         encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
        ...         on = ["title", "article"],
        ...         k = 10,
        ...    )
        ...
        ...    retriever = retrieve.Elastic(key="id", on=["title", "article"], k=2, index="test")
        ...    retriever = retriever.reset()
        ...    retriever = retriever.add_embeddings(documents=documents, ranker=ranker)
        ...
        ...    answers = retriever("Paris")
        ...    assert answers[0]["embedding"].shape == (768,)

        """
        if embeddings is None:
            embeddings = ranker.embs(documents=documents)

        documents_embeddings = [
            {
                "_id": doc[self.key],
                "_index": self.index,
                "_source": {**doc, "embedding": embedding},
            }
            for doc, embedding in zip(documents, embeddings)
        ]
        helpers.bulk(self.es, documents_embeddings)
        self.es.indices.refresh(index=self.index)

        return self

    def add(self, documents: list) -> "Elastic":
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

        helpers.bulk(self.es, documents)
        self.es.indices.refresh(index=self.index)

        return self

    def __call__(self, q: str, query: str = None) -> list:
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
            # Returns stored embeddings as numpy array.
            if "embedding" in document:
                document["embedding"] = np.array(document["embedding"])
            ranked.append(document)

        return ranked

    def reset(self) -> "Elastic":
        """Delete the selected index from ElasticSearch."""
        if self.es.indices.exists(index=self.index):
            self.es.delete_by_query(index=self.index, body={"query": {"match_all": {}}})
            self.es.indices.refresh(index=self.index)
        return self
