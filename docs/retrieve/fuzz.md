# Fuzz

`retrieve.Fuzz` is a wrapper of [RapidFuzz](https://github.com/maxbachmann/RapidFuzz). It is a blazing fast library dedicated to fuzzy string matching. Documents can be indexed online with this retriever using the `add` method.

[RapidFuzz](https://github.com/maxbachmann/RapidFuzz) provides more scoring functions for the fuzzy string matching task. We can select the most suitable method for our dataset with the `fuzzer` parameter. The default scoring function is `fuzz.partial_ratio`.

```python
>>> from cherche import retrieve
>>> from rapidfuzz import fuzz

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

# List of available scoring function
>>> scoring = [
...     fuzz.ratio,
...     fuzz.partial_ratio,
...     fuzz.token_set_ratio,
...     fuzz.partial_token_set_ratio,
...     fuzz.token_sort_ratio,
...     fuzz.partial_token_sort_ratio,
...     fuzz.token_ratio,
...     fuzz.partial_token_ratio,
...     fuzz.WRatio,
...     fuzz.QRatio,
... ]

>>> retriever = retrieve.Fuzz(
...    key = "id",
...    on = ["title", "article"],
...    fuzzer = fuzz.partial_ratio, # Choose the scoring function.
... )

# Index documents
>>> retriever.add(documents)

>>> retriever("fashion", k=2)
[{'id': 1, 'similarity': 100.0}, {'id': 0, 'similarity': 46.15384615384615}]
```

## Batch retrieval

If we have several queries for which we want to retrieve the top k documents then we can
pass a list of queries to the retriever. In batch-mode, retriever returns a list of list of
documents instead of a list of documents.

```python
>>> retriever(["france", "arts", "capital"], k=30)
[[{'id': 0, 'similarity': 100.0}, # Match query 1
  {'id': 2, 'similarity': 100.0},
  {'id': 1, 'similarity': 66.66666666666667}],
 [{'id': 1, 'similarity': 100.0}, # Match query 2
  {'id': 0, 'similarity': 75.0},
  {'id': 2, 'similarity': 75.0}],
 [{'id': 0, 'similarity': 100.0}, # Match query 3
  {'id': 1, 'similarity': 44.44444444444444},
  {'id': 2, 'similarity': 44.44444444444444}]]
```

## Map keys to documents

We can map documents to retrieved keys.

```python
>>> retriever += documents
>>> retriever("fashion", k=30)
[{'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 100.0},
 {'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 46.15384615384615},
 {'id': 2,
  'article': 'The City of Paris is the centre and seat of government of the region and province of Île-de-France .',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 46.15384615384615}]
```
