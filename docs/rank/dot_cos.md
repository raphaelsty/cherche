# Similarity

Cherche provides two functions to measure the semantic similarity between a query and a document:
 `similarity.cosine` (higher is better) and `similarity.dot` (higher is better).

The choice of this function depends on the pre-trained model you are using for the ranking. If the
model has been trained with the cosine similarity then you should use `similarity.cosine`
otherwise if it has been trained with the dot product you should use `similarity.dot`.

These functions are only used by the `rank.Encoder` and `rank.DPR` models.

## Cosine

Initialization of `rank.Encoder` using cosine similarity:

```python
>>> from cherche import retrieve, rank, similarity
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
...    similarity = similarity.cosine,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker 
>>> search.add(documents)

>>> search(q="paris")
 [{'id': 0, 'similarity': 0.66051394}, {'id': 1, 'similarity': 0.5142564}]
```

## Dot

Initialization of `rank.DPR` using dot product:

```python
>>> from cherche import retrieve, rank, similarity
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

>>> ranker = rank.DPR(
...    on = "article",
...    key = "id",
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    k = 30,
...    similarity = similarity.dot,
...    path = "dpr.pkl"
... )

>>> search = retriever + ranker 
>>> search.add(documents)

>>> search(q="paris")
[{'id': 0, 'similarity': 75.669365},
 {'id': 1, 'similarity': 74.356224},
 {'id': 2, 'similarity': 72.9366}]
```
