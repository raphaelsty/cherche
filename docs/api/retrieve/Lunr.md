# Lunr

Lunr is a Python implementation of Lunr.js by Oliver Nightingale. Lunr is a retriever dedicated for small and middle size corpus.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **documents** (*list*)

    Documents in Lunr retriever are static. The retriever must be reseted to index new documents.

- **k** (*Optional[int]*) – defaults to `None`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Eiffel tower"},
...     {"id": 1, "title": "Paris", "article": "Paris is in France."},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada."},
... ]

>>> retriever = retrieve.Lunr(
...     key="id",
...     on=["title", "article"],
...     documents=documents,
... )

>>> retriever
Lunr retriever
    key      : id
    on       : title, article
    documents: 3

>>> print(retriever(q="paris", k=2))
[{'id': 1, 'similarity': 0.268}, {'id': 0, 'similarity': 0.134}]

>>> print(retriever(q=["paris", "montreal"], k=2))
[[{'id': 1, 'similarity': 0.268}, {'id': 0, 'similarity': 0.134}],
 [{'id': 2, 'similarity': 0.94}]]
```

## Methods

???- note "__call__"

    Retrieve documents from the index.

    **Parameters**

    - **q**     (*Union[str, List[str]]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
## References

1. [Lunr.py](https://github.com/yeraydiazdiaz/lunr.py)
2. [Lunr.js](https://lunrjs.com)
2. [Solr](https://solr.apache.org)

