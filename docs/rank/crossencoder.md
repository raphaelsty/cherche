# rank.CrossEncoder

The `rank.CrossEncoder` model re-ranks documents in ouput of the retriever using a pre-trained [CrossEncoder](https://huggingface.co/cross-encoder). The cross coder takes as input both the query and the document and produces a score accordingly. The `rank.Encoder` can't pre-compute embeddings to speed up search. A GPU will significantly speed up the cross-encoder.

## Requirements

To use the CrossEncoder ranker we will need to install "cherche[cpu]"

```sh
pip install "cherche[cpu]"
```

or on GPU:

```sh
pip install "cherche[gpu]"
```

## Documents

The cross-encoder must take as input the content of the document and not only the keys.

```python
search = retriever + documents + cross_encoder
```

## Tutorial

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import CrossEncoder

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

# Cross-Encoder needs documents contents. So we map documents to the output of retriever.
>>> retriever += documents

>>> ranker = rank.CrossEncoder(
...     on = ["title", "article"],
...     encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").predict,
... )

>>> match = retriever(["paris", "art", "fashion"], k=100)

# Re-rank output of retriever
>>> ranker(["paris", "art", "fashion"], documents=match, k=30)
[[{'id': 0, # Query 1
   'article': 'Paris is the capital and most populous city of France',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 6.915566},
  {'id': 2,
   'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 6.651541},
  {'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 4.3157015}],
 [{'id': 1, # Query 2
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': -2.9981978}],
 [{'id': 2, # Query 3
   'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': -4.356096},
  {'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': -4.7529125}]]
```

## Ranker in pipeline

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import CrossEncoder

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

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=100)

>>> ranker = rank.CrossEncoder(
...     on = ["title", "article"],
...     encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").predict,
... )

# Cross-Encoder needs documents contents. So we map documents to the output of retriever.
>>> search = retriever + documents + ranker
>>> search(q=["paris", "arts", "fashion"])
[[{'id': 0, # Query 1
   'article': 'Paris is the capital and most populous city of France',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 6.915566},
  {'id': 2,
   'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 6.651541},
  {'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': 4.3157015}],
 [{'id': 1, # Query 2
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': -2.9981978}],
 [{'id': 2, # Query 3
   'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': -4.356096},
  {'id': 1,
   'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
   'title': 'Paris',
   'url': 'https://en.wikipedia.org/wiki/Paris',
   'similarity': -4.7529125}]]
```

## Pre-trained Cross-Encoders

Pre-trained Cross-Encoder are available on [Hugging Face hub](https://huggingface.co/cross-encoder).