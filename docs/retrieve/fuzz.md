# Fuzz

`retrieve.Fuzz` is a wrapper of [RapidFuzz](https://github.com/maxbachmann/RapidFuzz). It is a
blazzing fast library dedicated to fuzzy string matching. This retriever is very fast but the
documents are stored in memory. Documents can be indexed in an online fashion way with this
retriever using `add` method.

[RapidFuzz](https://github.com/maxbachmann/RapidFuzz) provides more scoring functions for the fuzzy
string matching task. You can select the most suitable method for your dataset with the `fuzzer`
parameter of `retrieve.Fuzz`. The default scoring function is `fuzz.partial_ratio`.

```python
>>> from cherche import retrieve
>>> from rapidfuzz import fuzz, string_metric

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
...     string_metric.levenshtein,
...     string_metric.normalized_levenshtein,
... ]

>>> retriever = retrieve.Fuzz(
...    key = "id", 
...    on = ["title", "article"], 
...    k = 2,
...    fuzzer = fuzz.partial_ratio, # Choose the scoring function.
... )

>>> retriever.add(documents)

>>> retriever("fashion")
[{'id': 1, 'similarity': 100.0}, {'id': 0, 'similarity': 46.15384615384615}]
```

## Map keys to documents

```python
>>> retriever += documents
>>> retriever("fashion")
[{'id': 1,
  'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 100.0},
 {'id': 0,
  'article': 'Paris is the capital and most populous city of France',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 46.15384615384615}]
```
