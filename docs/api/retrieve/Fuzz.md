# Fuzz

[RapidFuzz](https://github.com/maxbachmann/RapidFuzz) wrapper. Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*)

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **fuzzer** – defaults to `<cyfunction partial_ratio at 0x16700dad0>`

    [RapidFuzz scorer](https://maxbachmann.github.io/RapidFuzz/Usage/fuzz.html): fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio, fuzz.partial_token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_token_sort_ratio, fuzz.token_ratio, fuzz.partial_token_ratio, fuzz.WRatio, fuzz.QRatio, string_metric.levenshtein, string_metric.normalized_levenshtein

- **default_process** (*bool*) – defaults to `True`

    Pre-processing step. If set to True, documents processed by [RapidFuzz default process.](https://maxbachmann.github.io/RapidFuzz/Usage/utils.html)


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from rapidfuzz import fuzz

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki", "tags": ["paris", "capital"]},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki", "tags": ["canada", "montreal"]},
... ]

>>> retriever = retrieve.Fuzz(
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
...    fuzzer = fuzz.partial_ratio,
... )

>>> retriever.add(documents=documents)
Fuzz retriever
     key: id
     on: title, article
     documents: 3
     fuzzer: partial_ratio

>>> retriever("Paris")
[{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "Paris", "author": "Wiki", "tags": ["paris", "capital"]},
...    {"id": 1, "title": "Eiffel tower", "article": "Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
...    {"id": 2, "title": "Montreal", "article": "Canada.", "author": "Wiki", "tags": ["canada", "montreal"]},
... ]

>>> retriever.add(documents=documents)
Fuzz retriever
     key: id
     on: title, article
     documents: 3
     fuzzer: partial_ratio

>>> retriever("Paris")
[{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

>>> documents = [
...    {"id": 3, "title": "Paris", "article": "Paris", "author": "Wiki", "tags": ["paris", "capital"]},
...    {"id": 4, "title": "Paris", "article": "Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
... ]

>>> retriever.add(documents = documents)
Fuzz retriever
     key: id
     on: title, article
     documents: 5
     fuzzer: partial_ratio

>>> retriever("Paris")
[{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]
```

## Methods

???- note "__call__"

    Retrieve documents using Fuzz.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

    Fuzz is streaming friendly.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
## References

1. [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

