__all__ = ["Milvus"]

import numpy as np


class Milvus:
    """Milvus index.

    Parameters
    ----------
    collection
        Milvus collection.
    vector_field
        Field of the Milvus collection dedicated to embeddings.
    search_params
        Milvus search [parameters](https://milvus.io/docs/v2.0.0/index.md) which are specific to the index.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import index, retrieve
    >>> from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...    {"id": 0, "title": "Paris"},
    ...    {"id": 1, "title": "Madrid"},
    ...    {"id": 2, "title": "Paris"}
    ... ]

    >>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    >>> encoder.encode(["test"]).shape[1]
    768

    >>> connections.connect(
    ...    alias="default",
    ...    host='localhost',
    ...    port='19530'
    ... )

    >>> key = FieldSchema(
    ...    name="id",
    ...    dtype=DataType.INT64,
    ...    is_primary=True,
    ... )

    >>> title = FieldSchema(
    ...    name="title",
    ...    dtype=DataType.VARCHAR,
    ...    max_length=200,
    ... )

    >>> embedding = FieldSchema(
    ...    name="embeddings",
    ...    dtype=DataType.FLOAT_VECTOR,
    ...    dim = 768,
    ... )

    >>> schema = CollectionSchema(
    ...    fields=[key, title, embedding], description="Test."
    ... )

    >>> collection = Collection(
    ...    name="documentation",
    ...    schema=schema,
    ...    using='default',
    ...    shards_num=2,
    ... )

    >>> _ = collection.create_index(
    ...    field_name = "embeddings",
    ...    index_params = {
    ...        "metric_type": "L2",
    ...        "index_type": "IVF_FLAT",
    ...        "params": {"nlist": 1024}
    ...     }
    ... )

    >>> milvus = index.Milvus(
    ...     collection=collection,
    ...     vector_field="embeddings",
    ...     search_params={"metric_type": "L2", "params": {"nprobe": 10}},
    ... )

    >>> milvus = milvus.add(
    ...     documents=documents,
    ...     embeddings=encoder.encode(["Paris", "Madrid", "Paris"]),
    ... )

    >>> results = milvus(embedding = encoder.encode(["Spain"]), key="id")

    >>> print(results)
    [{'id': 1, 'similarity': 1.5076334135501044, 'title': 'Madrid'},
     {'id': 2, 'similarity': 0.9021741164485997, 'title': 'Paris'},
     {'id': 0, 'similarity': 0.9021741164485997, 'title': 'Paris'}]

    >>> utility.drop_collection("documentation")

    >>> collection = Collection(
    ...    name="documentation",
    ...    schema=schema,
    ...    using='default',
    ...    shards_num=2,
    ... )

    >>> _ = collection.create_index(
    ...    field_name = "embeddings",
    ...    index_params = {
    ...        "metric_type": "L2",
    ...        "index_type": "IVF_FLAT",
    ...        "params": {"nlist": 1024}
    ...     }
    ... )

    >>> milvus = index.Milvus(
    ...     collection=collection,
    ...     vector_field="embeddings",
    ...     search_params={"metric_type": "L2", "params": {"nprobe": 10}},
    ... )

    >>> retriever = retrieve.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    key = "id",
    ...    on = ["title", "author"],
    ...    k = 2,
    ...    index = milvus
    ... )

    >>> retriever = retriever.add(documents)

    >>> print(retriever("spain"))
    [{'id': 1, 'similarity': 1.5076334135501044, 'title': 'Madrid'},
     {'id': 0, 'similarity': 0.9021741164485997, 'title': 'Paris'}]

    >>> utility.drop_collection("documentation")

    >>> collection = Collection(
    ...    name="documentation",
    ...    schema=schema,
    ...    using='default',
    ...    shards_num=2,
    ... )

    >>> _ = collection.create_index(
    ...    field_name = "embeddings",
    ...    index_params = {
    ...        "metric_type": "L2",
    ...        "index_type": "IVF_FLAT",
    ...        "params": {"nlist": 1024}
    ...     }
    ... )

    >>> milvus = index.Milvus(
    ...     collection=collection,
    ...     vector_field="embeddings",
    ...     search_params={"metric_type": "L2", "params": {"nprobe": 10}},
    ... )

    >>> retriever = retrieve.DPR(
    ...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
    ...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
    ...    key = "id",
    ...    on = ["title", "author"],
    ...    k = 2,
    ...    index = milvus
    ... )

    >>> retriever = retriever.add(documents)

    >>> print(retriever("spain"))
    [{'id': 1, 'similarity': 0.008724426096678638, 'title': 'Madrid'},
     {'id': 0, 'similarity': 0.0066852363141677515, 'title': 'Paris'}]

    >>> utility.drop_collection("documentation")

    References
    ----------
    1. [Collections](https://milvus.io/docs/v2.1.x/create_collection.md)
    2. [Indexes](https://milvus.io/docs/v2.1.x/build_index.md)

    """

    def __init__(self, collection, vector_field: str, search_params: dict):
        self.collection = collection
        self.vector_field = vector_field
        self.search_params = search_params
        self.output_fields = [
            field["name"]
            for field in self.collection.schema.to_dict()["fields"]
            if field["name"] != self.vector_field
        ]

    def __len__(self) -> int:
        return self.collection.num_entities

    def add(self, documents: list, embeddings: list, **kwargs) -> "Milvus":
        data = []
        for field in self.collection.schema.to_dict()["fields"]:
            if field["name"] == self.vector_field:
                data.append(embeddings)
            else:
                data.append([document.get(field["name"], "") for document in documents])

        self.collection.insert(data)
        return self

    def __call__(
        self,
        embedding: np.ndarray,
        key: str,
        k: int = None,
        expr: str = None,
        consistency_level: str = None,
        partition_names: list = None,
        **kwargs
    ) -> list:
        """Retrieve documents

        Parameters
        ----------
        embedding
            Document embedding.
        k
            Number of documents to retrieve.
        expr
            Searching with predicates.
        consistency_level
            Consistency level of the search.
        partition_names
            List of names of the partition to search in.
        """
        self.collection.load()

        if k is None:
            k = len(self)

        q = {
            "data": embedding,
            "anns_field": self.vector_field,
            "param": self.search_params,
            "limit": k,
            "output_fields": self.output_fields,
        }

        if expr is not None:
            q["expr"] = expr

        if consistency_level is not None:
            q["consistency_level"] = consistency_level

        if partition_names is not None:
            q["partition_names"] = partition_names

        match = self.collection.search(**q)[0]

        return [
            {
                **{key: key, "similarity": 1 / distance if distance > 0 else 0},
                **{field: fields.entity.get(field) for field in self.output_fields},
            }
            for fields, distance in zip(match, match.distances)
        ]
