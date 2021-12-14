# Flash

FlashText Retriever. Flash aims to find documents that contain keywords such as a list of tags for example.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **keywords** (*flashtext.keyword.KeywordProcessor*) – defaults to `None`

    Keywords extractor from [FlashText](https://github.com/vi3k6i5/flashtext). If set to None, a default one is created.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.Flash(on="tags", k=2)

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki", "tags": ["paris", "capital"]},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki", "tags": ["paris", "eiffel", "tower"]},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki", "tags": ["canada", "montreal"]},
... ]

>>> retriever = retriever.add(documents=documents)

>>> print(retriever(q="paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'tags': ['paris', 'capital'],
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'tags': ['paris', 'eiffel', 'tower'],
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Retrieve tagss.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add keywords to the retriever.

    **Parameters**

    - **documents**     (*list*)    
    
## References

1. [FlashText](https://github.com/vi3k6i5/flashtext)
2. [Replace or Retrieve Keywords In Documents at Scale](https://arxiv.org/abs/1711.00046)

