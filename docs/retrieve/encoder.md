# Encoder

The `retriever.Encoder` is useful when you want to retrieve documents based on the semantic similarity of the query. The `retriever.Encoder` model encodes queries and documents within a single model. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) and [Hugging Face similarity](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) models. Documents indexed by `retrieve.Encoder` can be updated in mini-batch with the `add` method. This method takes time because the encoder pre-computes the document embeddings and stores them in the index.

## Index

The index of the retriever.Encoder allows to speed up the search for the nearest neighbor. The index is stopped in memory when we use Faiss. It is stored on disk when we use Milvus.

The `retriever.Encoder` uses [Faiss](https://github.com/facebookresearch/faiss) to store pre-computed document embeddings. However, if we work with a large corpus, i.e., several million records, we can replace the Faiss index with a [Milvus](https://milvus.io/docs/v2.1.x/overview.md) index. Milvus is a vector-oriented database that allows storing all the vectors on the disk rather than in memory.

### Faiss Index

By default, the index used is the `IndexFlatL2`. It is stored in memory and is called via the CPU. Faiss offers various algorithms suitable for different corpus sizes and speed constraints. [Here are the guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

```python
>>> from cherche import retrieve
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

>>> retriever = retrieve.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 2,
... )

>>> retriever = retriever.add(documents=documents)

>>> retriever("paris")
```

```python
[{'id': 0, 'similarity': 0.9025790931437582},
 {'id': 2, 'similarity': 0.8160134832855334}]
```

Let's create a faiss index stored in memory that runs on GPU with the sentence transformer as an encoder that also runs on GPU.

```sh
pip install faiss-gpu
```

```python
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer
>>> import faiss

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

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

>>> d = encoder.encode("Embeddings size.").shape[0]
>>> index = faiss.IndexFlatL2(d)
# # 0 is the id of the GPU
>>> index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

>>> retriever = retrieve.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = encoder.encode,
...    k = 2,
...    index = index,
... )

>>> retriever.add(documents)

>>> retriever("paris")
```

```python
[{'id': 0,
  'similarity': 0.9025790931437582},
 {'id': 2,
  'similarity': 0.8160134832855334}]
```

#### Map keys to documents

```python
>>> retriever += documents
>>> retriever("france")
[{'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.9025790931437582},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.8160134832855334}]
```

### Milvus index

[Milvus](https://github.com/milvus-io/milvus) is an open-source vector database built to power embedding similarity search and AI applications.  When the Faiss index does not fit in the memory, it is suitable to use a Milvus index.

#### Milvus installation with Docker

Fetch the latest version of Milvus image:

```sh
wget https://github.com/milvus-io/milvus/releases/download/v2.1.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
```

Start the milvus container:

```sh
sudo docker-compose up -d
```

Stop the milvus container:

```sh
sudo docker-compose down
```

#### Collection

Before indexing the documents, it is necessary to create a collection. We will use the Milvus Python API: [pymilvus](https://milvus.io/api-reference/pymilvus/v2.1.1/About.md).

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

#### Milvus retriever

Once we have created our Milvus collection and index, we can initialize our `retriever.Encoder`, perform searches, and integrate it into a pipeline:

```python
>>> from cherche import retrieve, index
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

>>> retriever = retrieve.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title", "author"],
...    k = 2,
...    index = milvus
... )

>>> retriever = retriever.add(documents)

>>> retriever("spain")
```

```python
[{'id': 1, 'similarity': 1.5076334135501044, 'title': 'Madrid'},
 {'id': 0, 'similarity': 0.9021741164485997, 'title': 'Paris'}]
```

## Custom Encoder

We can use custom models within `retrieve.Encoder`. It should encode a list of documents `list[str]` and return a NumPy array with dimensions `(number of documents, embedding size)`. This model should also encode a query (str) and return an embedding of size `(embedding dimension, )`. For example, we could use word embeddings to encode documents and queries.

Here is an example of how to integrate a custom encoder:

```python
import numpy as np
import typing
from cherche import retrieve
from sentence_transformers import SentenceTransformer

class CustomEncoder:

    def __init__(self):
      """Custom Encoder retriever."""
      self.encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def encode(self, documents: typing.List[str]):
      """Documents encoder."""
      return self.encoder.encode(documents)

model = CustomEncoder()

# Your model should pass these tests, i.e Sentence Bert API.
assert model.encode(["Paris", "France", "Bordeaux"]).shape[0] == 3
assert isinstance(model.encode(["Paris", "France", "Bordeaux"]), np.ndarray)

assert len(model.encode("Paris").shape) == 1
assert isinstance(model.encode("Paris"), np.ndarray)

retriever = retrieve.Encoder(
    encoder = model.encode,
    key = "id",
    on = ["title", "article"],
    k = 2,
)
```
