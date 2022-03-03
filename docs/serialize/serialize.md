# Save & Load

Serialization in Python saves an object on the disk to reload it during a new session. Using Cherche, we could prototype a neural search pipeline in a notebook before deploying it on an API. We can also save a neural search pipeline to avoid recomputing embeddings of the ranker.

We must ensure that the package versions are identical in both environments (dumping and loading).

## Saving and loading on same environment

### Saving

We will initialize and save our pipeline in a `search.pkl` file

```python
>>> from cherche import data, retrieve, rank
>>> from sentence_transformers import SentenceTransformer
>>> import pickle

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 10,
... )

>>> search = retriever + ranker
# Pre-compute embeddings of the ranker
>>> search.add(documents=documents)

# Dump our pipeline using pickle.
# The file search.pkl contains our pipeline
>>> with open("search.pkl", "wb") as search_file:
...    pickle.dump(search, search_file)

```

### Loading

After saving our pipeline in the file `search.pkl`, we can reload it using Pickle.

```python
>>> import pickle

>>> with open("search.pkl", "rb") as search_file:
...    search = pickle.load(search_file)

>>> search("bordeaux")
[{'id': 57, 'similarity': 0.69513476},
 {'id': 63, 'similarity': 0.6214991},
 {'id': 65, 'similarity': 0.61809057},
 {'id': 59, 'similarity': 0.61285114},
 {'id': 71, 'similarity': 0.5893674},
 {'id': 67, 'similarity': 0.5893066},
 {'id': 74, 'similarity': 0.58757037},
 {'id': 61, 'similarity': 0.58593774},
 {'id': 70, 'similarity': 0.5854107},
 {'id': 66, 'similarity': 0.56525207}]
```

## Saving on GPU, loading on CPU

Typically, we could pre-compute the document integration on google collab with a GPU before
deploying our neural search pipeline on a CPU-based instance.

When transferring the pipeline that runs on the GPU to a machine that will run it on the CPU, it will be necessary to avoid serializing the `retrieve.Encoder`, `retrieve.DPR`, `rank.DPR` and `rank.Encoder`. These retrievers and rankers would not be compatible if we initialized them on GPU. We will have to replace the models on GPU to put them on CPU. We must ensure that the package versions are strictly identical in both environments (GPU and CPU).

### Saving on GPU

```python
>>> from cherche import data, retrieve, rank
>>> from sentence_transformers import SentenceTransformer
>>> import pickle

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda").encode,
...    k = 10,
... )

>>> search = retriever + ranker
# Pre-compute embeddings of the ranker
>>> search.add(documents=documents)

# Replace the GPU-based encoder with a CPU-based encoder. 
>>> ranker.encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode

with open("search.pkl", "wb") as search_file:
    pickle.dump(search, search_file)

```

### Loading on CPU

We can load our neural search pipeline using `pickle.load` in a new session.

```python
>>> import pickle

>>> with open("search.pkl", "rb") as search_file:
...    search = pickle.load(search_file)

>>> search("bordeaux")
[{'id': 57, 'similarity': 0.69513476},
 {'id': 63, 'similarity': 0.6214991},
 {'id': 65, 'similarity': 0.61809057},
 {'id': 59, 'similarity': 0.61285114},
 {'id': 71, 'similarity': 0.5893674},
 {'id': 67, 'similarity': 0.5893066},
 {'id': 74, 'similarity': 0.58757037},
 {'id': 61, 'similarity': 0.58593774},
 {'id': 70, 'similarity': 0.5854107},
 {'id': 66, 'similarity': 0.56525207}]
```
