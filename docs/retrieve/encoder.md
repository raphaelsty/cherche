## Encoder

To use the Encoder retriever we will need to install "cherche[cpu]"

```sh
pip install "cherche[cpu]"
```

The `retriever.Encoder` is a powerful tool when you need to retrieve documents based on semantic similarity of the query. It encodes both queries and documents within a single model. The `retriever.Encoder` model is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) and [Hugging Face similarity](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) models.

Using the `retriever.Encoder`, you can index your documents, and update them in mini-batches using the `add` method. However, note that the `add` method can take some time since the encoder pre-computes document embeddings and stores them in the index.

The `retriever.Encoder` uses [Faiss](https://github.com/facebookresearch/faiss) to store pre-computed document embeddings in an index. This index allows for fast nearest neighbor search.

By default, `retriever.Encoder` uses the `IndexFlatL2` index which is stored in memory and called via the CPU. However, Faiss offers various algorithms suitable for different corpus sizes and speed constraints. To choose the most suitable index for your use case, you can refer to [Faiss guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

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
...    normalize = True,
... )

>>> retriever = retriever.add(documents=documents)

>>> retriever("paris", k=30)
[{'id': 0, 'similarity': 0.5979780497084908},
 {'id': 2, 'similarity': 0.5571123271029782},
 {'id': 1, 'similarity': 0.5563819541294073}]
```

## Batch computation

If we have several queries for which we want to retrieve the top k documents then we can
pass a list of queries to the retriever. This is much faster for multiple queries. In batch-mode,
retriever returns a list of list of documents instead of a list of documents.

```python
>>> retriever(["paris", "arts"], k=30, batch_size=64)
[[{'id': 0, 'similarity': 0.5979780283951969}, # Match query 1
  {'id': 2, 'similarity': 0.5571123641024619},
  {'id': 1, 'similarity': 0.5563819725806741}],
 [{'id': 1, 'similarity': 0.38966597854511925}, # Match query 2
  {'id': 0, 'similarity': 0.36300965990952766},
  {'id': 2, 'similarity': 0.356841141737425}]]
```


## Run encoder on GPU

Let's create a faiss index stored in memory that runs on GPU with the sentence transformer as an encoder that also runs on GPU.

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
>>> d = encoder.encode("Embeddings size.").shape[0]
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

>>> retriever = retrieve.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = encoder.encode,
...    index = index, # Our index running on GPU.
...    normalize = True,
... )

>>> retriever.add(documents)

>>> retriever(["paris", "arts"], k=30, batch_size=64)
[[{'id': 0, 'similarity': 0.5979780283951969}, # Match query 1
  {'id': 2, 'similarity': 0.5571123641024619},
  {'id': 1, 'similarity': 0.5563819725806741}],
 [{'id': 1, 'similarity': 0.38966597854511925}, # Match query 2
  {'id': 0, 'similarity': 0.36300965990952766},
  {'id': 2, 'similarity': 0.356841141737425}]]
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
