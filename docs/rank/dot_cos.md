# Distance

Cherche provides two functions to measure the semantic similarity between a query and a document: `distance.cosine_distance` (lower is better) and `distance.dot_similarity` (higher is better).

The choice of this function depends above all on the pre-trained model you are using for the ranking. If the model has been trained with the cosine similarity then you should use `distance.cosine_distance` otherwise if it has been trained with the dot product you should use `distance.cosine_distance`.

These functions are only used by the `rank.Encoder` and `rank.DPR` models.

Initialization of a `rank.Encoder` model with a model trained using cosine similarity:

```python
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"document": "Lorem ipsum dolor sit amet"},
...    {"document": " Duis aute irure dolor in reprehenderit"},
... ]

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer(f"sentence-transformers/all-mpnet-base-v2").encode,
...    on = "document",
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
...    {"document": "Lorem ipsum dolor sit amet"},
...    {"document": " Duis aute irure dolor in reprehenderit"},
... ]

>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    on = "document",
...    k = 30,
...    distance = distance.dot_similarity,
...    path = "dpr.pkl"
... )
```
