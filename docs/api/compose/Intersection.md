# Intersection

Intersection gathers retrieved documents from multiples retrievers and ranked documents from multiples rankers only if they appears in all proposed documents.



## Parameters

- **models** (*list*)



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"title": "Paris", "article": "Paris is the capital of France", "author": "Wiki"},
...     {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris.", "author": "Wiki"},
...     {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> search = retrieve.TfIdf(on="title") & retrieve.TfIdf(on="article") & retrieve.TfIdf(on="author")

>>> search.add(documents)
Intersection
-----
TfIdf retriever
     on: title
     documents: 3
TfIdf retriever
     on: article
     documents: 3
TfIdf retriever
     on: author
     documents: 3
-----

>>> print(search(q = "Wiki Paris"))
[{'article': 'Paris is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'}]

>>> print(search(q = "Paris"))
[]

>>> print(search(q = "Wiki Paris Montreal Eiffel"))
[{'article': 'Paris is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'},
 {'article': 'Montreal is in Canada.', 'author': 'Wiki', 'title': 'Montreal'},
 {'article': 'Eiffel tower is based in Paris.',
  'author': 'Wiki',
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
???- note "add"

