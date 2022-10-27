# Rank

Rankers are models that measure the semantic similarity between a document and a query. Rankers filter out documents based on the semantic similarity between the query and the documents. Rankers are compatible with all the retrievers.

## key, on and k parameters

The `key` parameter is mandatory for `ranker.Encoder` and `ranker.DPR` but not needed for `ranker.ZeroShot`. The `key` parameter is the unique identifier of the documents in the corpus. We can use ranker on multiple fields with the `on` parameter. Rankers will concatenate selected fields to calculate the embeddings of the documents. The k parameter of ranker allows selecting the number of documents to keep after the ranking. By default, rankers will reorder documents without dropping any.

|      Ranker     | Precomputing |                                                          GPU                                                          |
|:---------------:|:------------:|:---------------------------------------------------------------------------------------------------------------------:|
|  ranker.Encoder |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
|    ranker.DPR   |       ✅      | Highly recommended when precomputing <br>embeddings if the corpus is large. <br>Not needed anymore after precomputing |
| ranker.ZeroShot |       ❌      |                     Highly recommended since <br>ranker.ZeroShot cannot precompute <br>embeddings                     |

The `rank.Encoder` and `rank.DPR` rankers pre-compute the document embeddings once for all with the `add` method. This step can be time-consuming if we do not have a GPU. The embeddings are pre-computed so that the model can then rank the retriever documents at lightning speed.

## Quick start

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "id": 0,
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 1,
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "id": 2,
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
... )

>>> search = retriever + ranker
>>> search.add(documents)
>>> search(q="france")
[{'id': 0, 'similarity': 0.44967225}, {'id': 2, 'similarity': 0.3609671}]
```

## Index

The embeddings pre-computed by the ranker via the `add` method are stored in memory by default. However, storing the embeddings in a Milvus database is possible in case all the document embeddings would saturate the memory. [Milvus](https://github.com/milvus-io/milvus) is an open-source vector database built to power embedding similarity search and AI applications. When the Faiss index does not fit in the memory, it is suitable to use a Milvus index.

### Milvus installation with Docker

Fetch the latest version of Milvus image:

```sh
wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

Start the milvus container:

```sh
sudo docker-compose up -d
```

Stop the milvus container

```sh
sudo docker-compose down
```

### Collection

Before pre-computing documents embeddings, it is necessary to create a collection. We will use the Milvus Python API: [pymilvus](https://milvus.io/api-reference/pymilvus/v2.1.1/About.md).

First we will install the python client pymilvus.

```sh
pip install pymilvus
```

The first step is to connect to the Milvus database:

```python
>>> from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

>>> connections.connect(
...    alias="default",
...    host='localhost',
...    port='19530'
... )
```

We will then create a collection with three fields dedicated to storing our documents with the embeddings of the records. The first field, `id`, is the unique document identifier (is_primary=True). The second field is dedicated to the document title. The third and last field is dedicated to our embeddings of dimension 768. The size of the embeddings can vary depending on the encoder we use.
We can find more information about creating collections in the Milvus [documentation](https://milvus.io/docs/v2.1.x/create_collection.md).

After creating the collection, we will create an index to be able to search. Milvus provides detailed documentation with various indexes for different needs: [here](https://milvus.io/docs/v2.1.x/build_index.md).

```python
>>> from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

>>> connections.connect(
...    alias="default",
...    host='localhost',
...    port='19530'
... )

>>> documents = [
...    {"id": 0, "title": "Paris"},
...    {"id": 1, "title": "Madrid"},
...    {"id": 2, "title": "Paris"}
... ]

>>> collection_name, vector_field = "documentation", "embeddings"

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
...    name=vector_field,
...    dtype=DataType.FLOAT_VECTOR,
...    dim = 768,
... )

>>> schema = CollectionSchema(
...    fields=[key, title, embedding], description="Test."
... )

>>> collection = Collection(
...    name=collection_name,
...    schema=schema,
...    using='default',
...    shards_num=2,
... )

# Creation of the index
>>> _ = collection.create_index(
...    field_name = "embeddings",
...    index_params = {
...        "metric_type": "L2",
...        "index_type": "IVF_FLAT",
...        "params": {"nlist": 1024}
...     }
... )
```

#### Milvus ranker

Once we have created our Milvus collection and index, we can initialize our `rank.Encoder` or `rank.DPR` models:

```python
>>> from cherche import index, rank, retrieve
>>> from sentence_transformers import SentenceTransformer
>>> from pymilvus import connections, Collection

>>> documents = [
...    {"id": 0, "title": "Paris"},
...    {"id": 1, "title": "Madrid"},
...    {"id": 2, "title": "Paris"}
... ]

>>> connections.connect(
...    alias="default",
...    host='localhost',
...    port='19530'
... )

>>> collection_name, vector_field = "documentation", "embeddings"

>>> milvus = index.Milvus(
...     collection=Collection(name=collection_name),
...     vector_field=vector_field,
...     search_params={"metric_type": "L2", "params": {"nprobe": 10}},
... )

>>> retriever = retrieve.TfIdf(key="id", on=["title"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title"],
...    k = 5,
...    store = milvus # Milvus store
... )

>>> pipeline = retriever + ranker

>>> pipeline = pipeline.add(documents)

>>> pipeline("spain madrid")
```

```python
[{'id': 1, 'similarity': 0.8895982646702374}]
```