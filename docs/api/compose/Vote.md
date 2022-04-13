# Vote

Voting operator. Average of the similarity scores of the documents.



## Parameters

- **models** (*list*)

    List of models of the union.



## Examples

```python
>>> from cherche import compose, retrieve

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> search = (
...    retrieve.TfIdf(key="id", on="title", documents=documents, k=3) *
...    retrieve.TfIdf(key="id", on="article", documents=documents, k=3)
... )

>>> search
Vote
-----
TfIdf retriever
     key: id
     on: title
     documents: 3
TfIdf retriever
     key: id
     on: article
     documents: 3
-----

>>> search("paris eiffel")
[{'id': 1, 'similarity': 0.5216793798120436}, {'id': 0, 'similarity': 0.4783206201879563}]
```

## Methods

???- note "__call__"

    

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

???- note "reset"

