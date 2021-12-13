# Distance

Cherche provides two functions to measure the semantic similarity between a query and a document: `distance.cosine_distance` (lower is better) and `distance.dot_similarity` (higher is better).

The choice of this function depends above all on the pre-trained model you are using for the ranking. If the model has been trained with the cosine similarity then you should use `distance.cosine_distance` otherwise if it has been trained with the dot product you should use `distance.cosine_distance`.

These functions are only used by the `rank.Encoder` and `rank.DPR` models.

Initialization of a `rank.Encoder` model with a model trained using cosine similarity:

```python
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    on = "article",
...    k = 30,
...    distance = distance.cosine_distance,
...    path = "encoder.pkl"
... )

>>> ranker.add(documents)
```

Initialization of a `rank.DPR` model with a model trained using dot product:

```python
>>> from cherche import rank, distance
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    on = "article",
...    k = 30,
...    distance = distance.dot_similarity,
...    path = "dpr.pkl"
... )
```
