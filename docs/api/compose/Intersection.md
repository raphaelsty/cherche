# Intersection

Intersection gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers only if they are proposed by all models of the intersection pipeline.



## Parameters

- **models** (*list*)



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"id": 0, "title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
...     {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
...     {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> search = (
...    retrieve.TfIdf(key="id", on="title", documents=documents) &
...    retrieve.TfIdf(key="id", on="article", documents=documents) &
...    retrieve.Flash(key="id", on="author")
... ) + documents

>>> search.add(documents)
Intersection
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

>>> print(search(q = "Wiki Paris"))
[{'article': 'Paris is the capital of France',
    'author': 'Wiki',
    'id': 0,
    'title': 'Paris'}]

>>> print(search(q = "Paris"))
[]

>>> print(search(q = "Wiki Paris Montreal Eiffel"))
[{'article': 'Paris is the capital of France',
      'author': 'Wiki',
      'id': 0,
      'title': 'Paris'},
     {'article': 'Montreal is in Canada.',
      'author': 'Wiki',
      'id': 2,
      'title': 'Montreal'},
     {'article': 'Eiffel tower is based in Paris.',
      'author': 'Wiki',
      'id': 1,
      'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

???- note "reset"

