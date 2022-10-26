# Lunr

Lunr is a Python implementation of Lunr.js by Oliver Nightingale. Lunr is a retriever dedicated for small and middle size corpus.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **documents** (*list*)

    Documents in Lunr retriever are static. The retriever must be reseted to index new documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.Lunr(key="id", on=["title", "article"], documents=documents, k=3)

>>> retriever
Lunr retriever
     key: id
     on: title, article
     documents: 3

>>> print(retriever(q="paris"))
[{'id': 0, 'similarity': 0.524}, {'id': 1, 'similarity': 0.414}]

>>> retriever += documents

>>> print(retriever(q="paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.524,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.414,
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Retrieve the right document.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
## References

1. [Lunr.py](https://github.com/yeraydiazdiaz/lunr.py)
2. [Lunr.js](https://lunrjs.com)
2. [Solr](https://solr.apache.org)

