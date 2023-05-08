# DPR

The `retriever.DPR` model uses DPR-based models that encode queries and documents with two distinct models. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) and [Hugging Face similarity](https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=downloads) DPR models.

To use the DPR retriever we will need to install "cherche[cpu]"

```sh
pip install "cherche[cpu]"
```

The `retriever.DPR` uses [Faiss](https://github.com/facebookresearch/faiss) to store pre-computed document embeddings in an index. This index allows for fast nearest neighbor search.

By default, `retriever.DPR` uses the `IndexFlatL2` index which is stored in memory and called via the CPU. However, Faiss offers various algorithms suitable for different corpus sizes and speed constraints. To choose the most suitable index for your use case, you can refer to [Faiss guidelines](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index).

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

>>> retriever = retrieve.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    key = "id",
...    on = ["title", "article"],
...    normalize = True,
... )

>>> retriever = retriever.add(documents=documents)

>>> retriever(["paris", "arts and science"], k=3, batch_size=64)
[[{'id': 1, 'similarity': 0.6063913848352276},
  {'id': 0, 'similarity': 0.6021773868199615},
  {'id': 2, 'similarity': 0.5844722795720981}],
 [{'id': 1, 'similarity': 0.5060106120613739},
  {'id': 0, 'similarity': 0.4877345511626579},
  {'id': 2, 'similarity': 0.4864927436178843}]]
```

## Run DPR on GPU

Let's create a faiss index stored in memory that runs on GPU with an encoder that also run on GPU.
The retriever will run much faster.

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

>>> d = encoder.encode("Embeddings size.").shape[0] # embeddings_dim
>>> index = faiss.IndexFlatL2(d)
# # 0 is the id of the GPU
>>> index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

>>> retriever = retrieve.DPR(
...    encoder = encoder.encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base', device="cuda").encode,
...    key = "id",
...    on = ["title", "article"],
...    normalize = True,
... )

>>> retriever.add(documents)

>>> retriever("paris")
[{'id': 0,
  'similarity': 0.9025790931437582},
 {'id': 2,
  'similarity': 0.8160134832855334}]
```

#### Map keys to documents

```python
>>> retriever += documents
>>> retriever(["france", "arts"], k=1, batch_size=64)
[[{'id': 0,
   'article': 'Paris is the capital and most populous city of France',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.5816614494510223}],
 [{'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 0.5130107727511707}]]
```
