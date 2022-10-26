# Union

Union gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers.



## Parameters

- **models** (*list*)

    List of models of the union.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> search = (
...     retrieve.TfIdf(key="id", on="title", documents=documents) |
...     retrieve.TfIdf(key="id", on="article", documents=documents) |
...     retrieve.Flash(key="id", on="author")
... ) + documents

>>> search.add(documents)
Union
-----
TfIdf retriever
     key: id
     on: title
     documents: 3
TfIdf retriever
     key: id
     on: article
     documents: 3
Flash retriever
     key: id
     on: author
     documents: 1
-----
Mapping to documents

>>> print(search(q = "Paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.3333333333333333,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.3333333333333333,
  'title': 'Eiffel tower'}]

>>> print(search(q = "Montreal"))
[{'article': 'Montreal is in Canada.',
  'author': 'Wiki',
  'id': 2,
  'similarity': 0.3333333333333333,
  'title': 'Montreal'}]

>>> print(search(q = "Wiki"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.1111111111111111,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.1111111111111111,
  'title': 'Eiffel tower'},
 {'article': 'Montreal is in Canada.',
  'author': 'Wiki',
  'id': 2,
  'similarity': 0.1111111111111111,
  'title': 'Montreal'}]
```

## Methods

???- note "__call__"

    

    **Parameters**

    - **q**     (*str*)     – defaults to ``    
    - **user**     (*Union[str, int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add new documents.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
???- note "reset"

