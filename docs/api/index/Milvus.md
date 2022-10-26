# Milvus

Milvus index.



## Parameters

- **collection**

    Milvus collection.

- **vector_field** (*str*)

    Field of the Milvus collection dedicated to embeddings.

- **search_params** (*dict*)

    Milvus search [parameters](https://milvus.io/docs/v2.0.0/index.md) which are specific to the index.

- **replica_number** – defaults to `1`

    Number of the replica to load.

- **partition_name** (*list*) – defaults to `None`

    Partition name to load.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import index, retrieve, utils
>>> from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
>>> from sentence_transformers import SentenceTransformer
>>> from implicit.als import AlternatingLeastSquares

>>> documents = [
...    {"id": 0, "title": "Paris"},
...    {"id": 1, "title": "Madrid"},
...    {"id": 2, "title": "Paris"}
... ]

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
>>> encoder.encode(["test"]).shape[1]
768

```

# Milvus index

```python
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

>>> print(milvus(embedding = encoder.encode(["Spain"]), key="id"))
[{'id': 1, 'similarity': 1.5076334135501044, 'title': 'Madrid'},
 {'id': 2, 'similarity': 0.9021741164485997, 'title': 'Paris'},
 {'id': 0, 'similarity': 0.9021741164485997, 'title': 'Paris'}]

>>> known, embeddings, unknown = milvus.get(key="id", values=[1, 2, 999])
>>> known
[1, 2]

>>> unknown
[999]

>>> utility.drop_collection("documentation")

```

# Retriever

```python
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

```

# Ranking

```python
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

```

# Collaborative filtering
#>>> utility.drop_collection("user")
#>>> utility.drop_collection("item")

```python
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
...    dim = 64,
... )

```

# Items Milvus collection
```python
>>> collection_items = Collection(
...    name="item",
...    schema=CollectionSchema(
...        fields=[key, title, embedding], description="Test."
...    ),
...    using='default',
...    shards_num=2,
... )

>>> _ = collection_items.create_index(
...    field_name = "embeddings",
...    index_params = {
...        "metric_type": "L2",
...        "index_type": "IVF_FLAT",
...        "params": {"nlist": 1024}
...     }
... )

```

# Milvus index documents
```python
>>> milvus_index = index.Milvus(
...     collection=collection_items,
...     vector_field="embeddings",
...     search_params={"metric_type": "L2", "params": {"nprobe": 10}},
... )

```

# Users Milvus collection
```python
>>> key = FieldSchema(
...    name="id",
...    dtype=DataType.INT64,
...    is_primary=True,
... )

>>> embedding = FieldSchema(
...    name="embeddings",
...    dtype=DataType.FLOAT_VECTOR,
...    dim = 64,
... )

>>> collection_users = Collection(
...    name="user",
...    schema=CollectionSchema(
...         fields=[key, embedding], description="Users."
...    ),
...    using='default',
...    shards_num=2,
... )

>>> _ = collection_users.create_index(
...    field_name = "embeddings",
...    index_params = {
...        "metric_type": "L2",
...        "index_type": "IVF_FLAT",
...        "params": {"nlist": 1024}
...     }
... )

```

# Milvus store users
```python
>>> milvus_store = index.Milvus(
...     collection=collection_users,
...     vector_field="embeddings",
...     search_params={"metric_type": "L2", "params": {"nprobe": 10}},
... )

>>> ratings = {
...    0: {0: 1, 1: 1},
...    1: {0: 1, 1: 2},
...    2: {2: 1},
...    3: {2: 1},
... }

>>> index_users, index_documents, sparse_ratings = utils.users_items_sparse(ratings=ratings)

>>> model = AlternatingLeastSquares(
...     factors=64,
...     regularization=0.05,
...     alpha=2.0,
...     iterations=100,
...     random_state=42,
... )

>>> model.fit(sparse_ratings)

>>> embeddings_users = {
...    user: embedding for user, embedding in zip(index_users, model.user_factors)
... }

>>> embeddings_documents = {
...    document: embedding
...    for document, embedding in zip(index_documents, model.item_factors)
... }

>>> recommend = retrieve.Recommend(
...    key="id",
...    k = 10,
...    index = milvus_index,
...    store = milvus_store,
... )

>>> recommend.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
...    embeddings_users=embeddings_users,
... )
Recommend retriever
    key: id
    Users: 4
    Documents: 3

>>> print(recommend(user=0))
[{'id': 1, 'similarity': 20113.03916599325, 'title': 'Madrid'},
 {'id': 0, 'similarity': 8998.286389012685, 'title': 'Paris'},
 {'id': 2, 'similarity': 0.41932083146673776, 'title': 'Paris'}]

>>> utility.drop_collection("user")
>>> utility.drop_collection("item")
```

## Methods

???- note "__call__"

    Retrieve documents

    **Parameters**

    - **embedding**     (*numpy.ndarray*)    
    - **key**     (*str*)    
    - **k**     (*int*)     – defaults to `None`    
    - **expr**     (*str*)     – defaults to `None`    
    - **consistency_level**     (*str*)     – defaults to `None`    
    - **partition_names**     (*list*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

???- note "get"

    Extract specific documents from their ids.

    **Parameters**

    - **values**     (*list*)    
    - **key**     (*str*)     – defaults to `None`    
    - **kwargs**    
    
## References

1. [Collections](https://milvus.io/docs/v2.1.x/create_collection.md)
2. [Indexes](https://milvus.io/docs/v2.1.x/build_index.md)

