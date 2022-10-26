# Intersection

Intersection gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers only if they are proposed by all models of the intersection pipeline.



## Parameters

- **models** (*list*)

    List of models of the union.



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
  'similarity': 0.6098051865303775,
  'title': 'Paris'}]

>>> print(search(q = "Paris"))
[]

>>> print(search(q = "Wiki Paris Montreal Eiffel"))
[{'article': 'Montreal is in Canada.',
  'author': 'Wiki',
  'id': 2,
  'similarity': 0.3423964772770161,
  'title': 'Montreal'},
 {'article': 'Paris is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.3215535643422896,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.33604995838069424,
  'title': 'Eiffel tower'}]
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

