# Rank

Rankers are models that measure the semantic similarity between a document and a query. The ranker allows to reorder the documents retrieved by the retriever (Tf-Idf, BM25, ...) based on the semantic similarity between the query and the documents retrieved.

The `rank.Encoder` and `rank.DPR` rankers pre-compute the document embeddings once for all with the `add` method. This step can be time consuming if you don't have a GPU. The embeddings are pre-computed so that the model can then rank the retriever documents at lightning speed. The embeddings can be saved in `pickle` format via the `path` parameter when the ranker is initialized. At a new initialization the model will use the pre-computed embeddings if the `path` parameter is provided.

## k and on parameters

The rankers all have a `k`-parameter during the initialization which allows to select the number of documents to keep after the ranking. The default value is`None`, i.e the ranker will not drop any documents.

The `on` parameter allows the ranker to be used on multiple fields. The rankers will concatenate the fields to calculate the embeddings of the documents.

## rank.Encoder

The `ranker.Encoder` model allows the use of framework that encode queries and documents with a single model. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) models.

You can use your own model within `ranker.Encoder`. This model should have an API similar to the Sentence Transformers models. It should have a method which encodes a list of documents `list[str]` which returns a numpy array with dimensions `(number of documents, embedding size)`. This same method must be able to encode a query (str) and return an embedding of size `(1, embedding dimension)`.

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
...    on = ["title", "article"],
...    k = 2,
...    path = "encoder.pkl"
... )

>>> ranker.add(documents)

>>> ranker(q="Paris", documents=documents)
```

```python
[{'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.6734946},
 {'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.61566335}]
```

## rank.DPR

The `ranker.DPR` model allows the use of framework that encode queries and documents with two distinct models. It is compatible with the [SentenceTransformers](https://www.sbert.net/docs/pretrained_models.html) DPR models.

You can use your own models within `ranker.DPR`. Theses models should have an API similar to the Sentence Transformers models. The document encoder should have a method which encodes a list of documents `list[str]` which returns a numpy array with dimensions `(number of documents, embedding size)`. The query encoder must be able to encode a query (str) and return an embedding of size `(1, embedding dimension)`.

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

>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    on = ["title", "article"],
...    k = 2,
...    path = "dpr.pkl"
... )

>>> ranker.add(documents)

>>> ranker(q="Paris", documents=documents)
```

```python
[{'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 75.669365},
 {'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 74.356224}]
```

## rank.ZeroShot

The `ranker.ZeroShot` model allows to use the `zero-shot-classification` pipeline of [Hugging Face](https://huggingface.co/facebook/bart-large-mnli). You can find more details on the zero-shot classification task in this very good [blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html). The `ranker.ZeroShot` model is slow because there is no pre-computation possible. It is therefore recommended to use this ranker with a GPU.

```python
>>> from cherche import rank
>>> from transformers import pipeline

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

>>> ranker = rank.ZeroShot(
...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
...     on = ["title", "article"],
...     k = 2,
... )

>>> ranker.add(documents)

>>> ranker(q="Paris", documents=documents)
```

```python
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.8914178609848022},
 {'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.6203346252441406}]
```

## Pre-compute embeddings

The `ranker.Encoder` and `ranker.DPR` models allow the pre-computation of document embeddings to speed up the search. These rankers will then store a mapping between the documents and its embedding.

When searching, the rankers will check if there are any embeddings pre-computed for a document and will use that embedding rather than recalculating it to significantly speed up the search.

The method `add` allows to pre-compute embeddings of the documents:

```python
>>> ranker.add(documents)
```
