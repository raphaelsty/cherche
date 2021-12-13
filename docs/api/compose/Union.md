# Union

Union gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers.



## Parameters

- **models** (*list*)



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...     {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
...     {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> search = retrieve.TfIdf(on="title") | retrieve.TfIdf(on="article")  | retrieve.Flash(on="author")

>>> search.add(documents)
Union
-----
TfIdf retriever
     on: title
     documents: 3
TfIdf retriever
     on: article
     documents: 3
Flash retriever
     on: author
     documents: 1
-----

>>> print(search(q = "Paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'title': 'Eiffel tower'}]

>>> print(search(q = "Montreal"))
[{'article': 'Montreal is in Canada.', 'author': 'Wiki', 'title': 'Montreal'}]

>>> print(search(q = "Wiki"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'title': 'Eiffel tower'},
 {'article': 'Montreal is in Canada.',
  'author': 'Wiki',
  'title': 'Montreal'}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

