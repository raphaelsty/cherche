# Similarity

Cherche provides two functions to measure the semantic similarity between a query and a document: `similarity.cosine` (bigger is better) and `similarity.dot` (bigger is better).

The choice of this function depends on the pre-trained model we are using for the ranking. For example, if we train our model with the cosine similarity, we should use `similarity.cosine`. Otherwise, if trained with the dot product, we should use `similarity.dot`.

We only use these functions as part of the rankers `rank.Encoder` and `rank.DPR` models.

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
... )

>>> search = retriever + ranker
>>> search.add(documents)

>>> search(q="paris")
[{'id': 0, 'similarity': 75.669365},
 {'id': 1, 'similarity': 74.356224},
 {'id': 2, 'similarity': 72.9366}]
```
