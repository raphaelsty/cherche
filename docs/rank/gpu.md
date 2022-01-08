# Gpu

GPU are mandatory to pre-compute embeddings using `rank.Encoder` and `rank.DPR` if you have a lot
of documents unless you are patient. After having pre-computed documents, the GPU will not be
needed anymore.

GPU is always needed using `rank.ZeroShot` to obtain decent results since it cannot pre-compute
embeddings.

If you are using the GPU to pre-compute the embeddings, remember to save your embeddings by setting
the `path` parameter of the `rank.Encoder` and `rank.DPR` rankers. To use the pre-computed embeddings
in another session, you will just have to initialize a new `ranker` with the same parameter `path`
and finally add the documents with the `add` method to automatically retrieve embeddings.

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
...    path = "all-mpnet-base-v2.pkl"
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
...    path = "dpr.pkl"
... )

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> search = retriever + ranker

# Pre-compute embeddings using GPU.
>>> search.add(documents=documents)
```

## rank.ZeroShot

To use the `zero-shot-classification` models with a GPU, the `device` parameter must be specified. By default the parameter `device` is set to `-1` to run on cpu. You needs to set it as a positive integer that match your cuda device id to run it on GPU.

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
