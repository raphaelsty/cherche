# Similarity

Cherche provides two functions to measure the semantic similarity between a query and a document: `similarity.cosine` (higher is better) and `similarity.dot` (higher is better).

The choice of this function depends above all on the pre-trained model you are using for the ranking. If the model has been trained with the cosine similarity then you should use `similarity.cosine` otherwise if it has been trained with the dot product you should use `similarity.dot`.

These functions are only used by the `rank.Encoder` and `rank.DPR` models.

## Cosine

Initialization of a `rank.Encoder` model with a model trained using cosine similarity:

```python
>>> from cherche import rank, similarity
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
...    similarity = similarity.cosine,
...    path = "encoder.pkl"
... )

>>> ranker.add(documents)
```

```python
Encoder ranker
    on: article
    k: 30
    similarity: cosine
    embeddings stored at: encoder.pkl
  ```

```python
>>> ranker(q = "fashion and gastronomy", documents=documents)
```

```python
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.40531972},
 {'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.2509912},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.13647239}]
```

## Dot

Initialization of a `rank.DPR` model with a model trained using dot product:

```python
>>> from cherche import rank, similarity
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
...    similarity = similarity.dot,
...    path = "dpr.pkl"
... )

>>> ranker.add(documents)
```

```python
DPR ranker
    on: article
    k: 30
    similarity: dot
    embeddings stored at: dpr.pkl
```

```python
>>> ranker(q = "fashion and gastronomy", documents=documents)
```

```python
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 63.076904},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 58.065662},
 {'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 55.83678}]
```
