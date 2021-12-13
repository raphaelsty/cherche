# Pipeline

Search allows you to build a neural search pipeline easily. Search offers three operators
to build a pipeline.

- `+` Main pipeline operator to put a ranker after a retriever or to put a question answering;, summarization model after a retriever or a ranker.

- `|` Union operator to gather the output of multiples retrievers or multiples rankers.

- `&` Intersection operator to filter the output of multiples retrievers or multiples rankers based on their intersection.

It is not currently possible to perform the intersection or union of a ranker and a retriever. It is necessary to perform union and intersection operations between rankers and retrievers.

## Pipeline `+`

```python
>>> from cherche import rank, retrieve

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

>>> retriever = retrieve.TfIdf(on="article", k = 30)

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "article",
...    k = 3,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker

>>> search.add(documents)
TfIdf retriever
    on: article
    documents: 3
Encoder ranker
    on: article
    k: 3
    similarity: cosine
    embeddings stored at: encoder.pkl

>>> search("fashion")
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.2555525}]
```

## Union `|`

The union operator `|` is used to improve model recall by bringing together documents retrieved by multiple models and or on different fields.

```python
>>> from cherche import rank, retrieve

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

>>> retriever = retrieve.TfIdf(on="title", k = 30) | retrieve.TfIdf(on="article", k = 30)

>>> ranker = (
...    rank.Encoder(
...        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...        on = "article",
...        k = 3,
...        path = "encoder.pkl"
...    ) | rank.Encoder(
...        encoder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1").encode,
...        on = "title",
...        k = 3,
...        path = "encoder.pkl"
...    )
... )

>>> search = retriever + ranker

>>> search.add(documents)
Union
-----
TfIdf retriever
    on: title
    documents: 3
TfIdf retriever
    on: article
    documents: 3
-----
Union
-----
Encoder ranker
    on: article
    k: 3
    similarity: cosine
    embeddings stored at: encoder.pkl
Encoder ranker
    on: title
    k: 3
    similarity: cosine
    embeddings stored at: encoder.pkl
-----

>>> search("fashion")
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.2555525}]
```

## Intersection `&`

The intersection operator improves the precision of the model by filtering documents on the intersection of retrievers and or rankers. With the intersect operator the number of documents returned by the models will be inferior to the union.

```python
>>> from cherche import rank, retrieve

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

>>> retriever = retrieve.TfIdf(on="title", k = 30) & retrieve.TfIdf(on="article", k = 30)


>>> ranker = (
...    rank.Encoder(
...        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...        on = "article",
...        k = 3,
...        path = "encoder.pkl"
...    ) & rank.Encoder(
...        encoder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1").encode,
...        on = "title",
...        k = 3,
...        path = "encoder.pkl"
...    )
... )

>>> search = retriever + ranker

>>> search.add(documents)
Intersection
-----
TfIdf retriever
    on: title
    documents: 3
TfIdf retriever
    on: article
    documents: 3
-----
Intersection
-----
Encoder ranker
    on: article
    k: 3
    similarity: cosine
    embeddings stored at: encoder.pkl
Encoder ranker
    on: title
    k: 3
    similarity: cosine
    embeddings stored at: encoder.pkl
-----

>>> search("fashion")
[]
```
