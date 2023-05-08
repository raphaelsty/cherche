# Flash

FlashText Retriever. Flash aims to find documents that contain keywords such as a list of tags for example.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **keywords** (*flashtext.keyword.KeywordProcessor*) – defaults to `None`

    Keywords extractor from [FlashText](https://github.com/vi3k6i5/flashtext). If set to None, a default one is created.

- **lowercase** (*bool*) – defaults to `True`

- **k** (*Optional[int]*) – defaults to `None`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...     {"id": 0, "title": "paris", "article": "eiffel tower"},
...     {"id": 1, "title": "paris", "article": "paris"},
...     {"id": 2, "title": "montreal", "article": "montreal is in canada"},
... ]

>>> retriever = retrieve.Flash(key="id", on=["title", "article"])

>>> retriever.add(documents=documents)
Flash retriever
    key      : id
    on       : title, article
    documents: 4

>>> print(retriever(q="paris", k=2))
[{'id': 1, 'similarity': 0.6666666666666666},
 {'id': 0, 'similarity': 0.3333333333333333}]

```

[{'id': 0, 'similarity': 1}, {'id': 1, 'similarity': 1}]

```python
>>> print(retriever(q=["paris", "montreal"]))
[[{'id': 1, 'similarity': 0.6666666666666666},
  {'id': 0, 'similarity': 0.3333333333333333}],
 [{'id': 2, 'similarity': 1.0}]]
```

## Methods

???- note "__call__"

    Retrieve documents from the index.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add keywords to the retriever.

    **Parameters**

    - **documents**     (*List[Dict[str, str]]*)    
    - **kwargs**    
    
## References

1. [FlashText](https://github.com/vi3k6i5/flashtext)
2. [Replace or Retrieve Keywords In Documents at Scale](https://arxiv.org/abs/1711.00046)

