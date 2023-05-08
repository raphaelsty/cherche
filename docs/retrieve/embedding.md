## Embedding

The `retriever.Embedding` is a powerful tool when you need to retrieve documents based on a custom model
or embeddings. The embeddings of documents must be of shape (n_documents, dim_embeddings). We can add
embeddings in a streaming fashion way (no need to add all documents embeddings at once).

To use the Embedding retriever we will need to install "cherche[cpu]"

```sh
pip install "cherche[cpu]"
```

The `retriever.Embedding` uses [Faiss](https://github.com/facebookresearch/faiss) to store pre-computed document embeddings in an index. This index allows for fast nearest neighbor search.

By default, `retriever.Embedding` uses the `IndexFlatL2` index which is stored in memory and called via the CPU. However, Faiss offers various algorithms suitable for different corpus sizes and speed constraints. To choose the most suitable index for your use case, you can refer to [Faiss guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

```python
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "id": 0,
...        "content": "Paris is the capital and most populous city of France",
...    },
...    {
...        "id": 1,
...        "content": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...    },
...    {
...        "id": 2,
...        "content": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...    }
... ]

# Let's use a custom encoder and create our documents embeddings of shape (n_documents, dim_embeddings)
>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
>>> embeddings_documents = encoder.encode([
...    document["content"] for document in documents
... ])

>>> retriever = retrieve.Embedding(
...    key = "id",
...    normalize = True,
... )

>>> retriever = retriever.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
... )

>>> queries = [
...    "Paris",
...    "Madrid",
...    "Montreal"
... ]

# Queries embeddings of shape (n_queries, dim_embeddings)
>>> embeddings_queries = encoder.encode(queries)
>>> retriever(embeddings_queries, k=2, batch_size=256)
[[{'id': 0, 'similarity': 0.5924658608170578},
  {'id': 1, 'similarity': 0.5446812754643415}],
 [{'id': 0, 'similarity': 0.40650240404418286},
  {'id': 1, 'similarity': 0.39610636961156953}],
 [{'id': 0, 'similarity': 0.42386890080792006},
  {'id': 2, 'similarity': 0.41253893705647166}]]
```

## Batch computation

If we have several queries for which we want to retrieve the top k documents then we can
pass a list of queries to the retriever. This is much faster for multiple queries. In batch-mode,
retriever returns a list of list of documents instead of a list of documents.

```python
>>> retriever(["paris", "arts"], k=30, batch_size=256)
[[{'id': 0, 'similarity': 0.5979780283951969}, # Match query 1
  {'id': 2, 'similarity': 0.5571123641024619},
  {'id': 1, 'similarity': 0.5563819725806741}],
 [{'id': 1, 'similarity': 0.38966597854511925}, # Match query 2
  {'id': 0, 'similarity': 0.36300965990952766},
  {'id': 2, 'similarity': 0.356841141737425}]]
```


## Run embedding on GPU

Let's create a faiss index stored in memory that runs on GPU with the sentence transformer as an encoder that also runs on GPU. The retriever will run much faster.

First we need to uninstall vanilla cherche and install "cherche[gpu]":

```sh
pip uninstall cherche
```

```sh
pip install "cherche[gpu]"
```

```python
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer
>>> import faiss

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
>>> d = encoder.encode("Embeddings size.").shape[0] # dim_embeddings
>>> index = faiss.IndexFlatL2(d)
>>> index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index) # 0 is the id of the GPU
```

```python
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

>>> retriever = retrieve.Embedding(
...    key = "id",
...    index = index, # Our index running on GPU.
... )

# Documents embeddings of shape (n_documents, dim_embeddings)
>>> embeddings_documents = encoder.encode([
...    document["content"] for document in documents
... ])

>>> retriever = retriever.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
... )

>>> queries = [
...    "Paris",
...    "Madrid",
...    "Montreal"
... ]

# Queries embeddings of shape (n_queries, dim_embeddings)
>>> embeddings_queries = encoder.encode(queries)
>>> retriever(embeddings_queries, k=2, batch_size=256)
[[{'id': 0, 'similarity': 0.5924658608170578},
  {'id': 1, 'similarity': 0.5446812754643415}],
 [{'id': 0, 'similarity': 0.40650240404418286},
  {'id': 1, 'similarity': 0.39610636961156953}],
 [{'id': 0, 'similarity': 0.42386890080792006},
  {'id': 2, 'similarity': 0.41253893705647166}]]
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
