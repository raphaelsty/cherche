# Fuzz

[RapidFuzz](https://github.com/maxbachmann/RapidFuzz) wrapper. Rapid fuzzy string matching in Python and C++ using the Levenshtein Distance.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **fuzzer** – defaults to `<cyfunction partial_ratio at 0x12fcc13c0>`

    [RapidFuzz scorer](https://maxbachmann.github.io/RapidFuzz/Usage/fuzz.html): fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio, fuzz.partial_token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_token_sort_ratio, fuzz.token_ratio, fuzz.partial_token_ratio, fuzz.WRatio, fuzz.QRatio, string_metric.levenshtein, string_metric.normalized_levenshtein

- **default_process** (*bool*) – defaults to `True`

    Pre-processing step. If set to True, documents processed by [RapidFuzz default process.](https://maxbachmann.github.io/RapidFuzz/Usage/utils.html)

- **k** (*Optional[int]*) – defaults to `None`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from rapidfuzz import fuzz

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 1, "title": "Paris", "article": "Paris is in France."},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada."},
... ]

>>> retriever = retrieve.Fuzz(
...    key = "id",
...    on = ["title", "article"],
...    fuzzer = fuzz.partial_ratio,
... )

>>> retriever.add(documents=documents)
Fuzz retriever
    key      : id
    on       : title, article
    documents: 3

>>> print(retriever(q="paris", k=2))
[{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}]

>>> print(retriever(q=["paris", "montreal"], k=2))
[[{'id': 0, 'similarity': 100.0}, {'id': 1, 'similarity': 100.0}],
 [{'id': 2, 'similarity': 100.0}, {'id': 1, 'similarity': 37.5}]]

>>> print(retriever(q=["unknown", "montreal"], k=2))
[[{'id': 2, 'similarity': 40.0}, {'id': 0, 'similarity': 36.36363636363637}],
 [{'id': 2, 'similarity': 100.0}, {'id': 1, 'similarity': 37.5}]]
```

## Methods

???- note "__call__"

    Retrieve documents from the index.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **tqdm_bar**     (*bool*)     – defaults to `True`    
    - **kwargs**    
    
???- note "add"

    Fuzz is streaming friendly.

    **Parameters**

    - **documents**     (*List[Dict[str, str]]*)    
    - **kwargs**    
    
## References

1. [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

