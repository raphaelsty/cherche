# Gpu

GPU is proper to pre-compute embeddings using `rank.Encoder` and `rank.DPR` if we have many documents. After having pre-computed documents, may GPU will not be needed anymore.

We strongly recommend using a GPU with`rank.ZeroShot` to obtain decent results since it cannot pre-compute embeddings.

## Rank.Encoder

```python
>>> from cherche import retrieve, rank
>>> from transformers import pipeline

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

# Ask the model to load and save embeddings at "all-mpnet-base-v2.pkl" to use the ranker on CPU.
>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2", device='cuda').encode,
...    k = 10,
... )

>>> search = retriever + ranker

# Pre-compute embeddings using GPU.
>>> search.add(documents=documents)
```

## Rank.DPR

```python
>>> from cherche import retrieve, rank
>>> from transformers import pipeline

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

# Ask the model to load and save embeddings at ./dpr.pkl
>>> ranker = rank.DPR(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base', device="cuda").encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base', devica="cuda").encode,
...    k = 10,
... )

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> search = retriever + ranker

# Pre-compute embeddings using GPU.
>>> search.add(documents=documents)
```

## rank.ZeroShot

We must set the `device` parameter to use GPU's `zero-shot-classification` models. The parameter `device` is set to -1 to run on the CPU by default. To run it on GPU, we need to set it as a positive integer that matches our Cuda device id.

```python
>>> from cherche import retrieve, rank
>>> from transformers import pipeline

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

>>> ranker = rank.ZeroShot(
...     on = ["title", "article"],
...     encoder = pipeline("zero-shot-classification",
...         model="typeform/distilbert-base-uncased-mnli",
...         device=0 # cuda:0, device=1 for cuda:1.
...     ),
...     k = 10,
... )

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

# ZeroShot needs documents
>>> search = retriever + documents + ranker

# Pre-compute embeddings using GPU.
>>> search.add(documents=documents)
```
