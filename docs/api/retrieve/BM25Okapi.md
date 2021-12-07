# BM25Okapi

BM25Okapi



## Parameters

- **on** (*str*)

- **tokenizer** – defaults to `None`

- **k** (*int*) – defaults to `None`

- **k1** – defaults to `1.5`

- **b** – defaults to `0.75`

- **epsilon** – defaults to `0.25`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.BM25Okapi(on="title", k=3, k1=1.5, b=0.75, epsilon=0.25)

>>> documents = [
...     {"url": "ckb/github.com", "title": "It is quite windy in London.", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch and Transformers .", "date": "22-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with Transformers .", "date": "22-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch and Transformers .", "date": "22-11-2021"},
... ]

>>> retriever = retriever.add(documents=documents)

>>> retriever
BM25Okapi retriever
    on: title
    documents: 5

>>> print(retriever(q="PyTorch Transformers"))
[{'date': '22-11-2021',
  'title': 'Github Library with PyTorch and Transformers .',
  'url': 'mkb/github.com'},
 {'date': '22-11-2021',
  'title': 'Github Library with PyTorch and Transformers .',
  'url': 'mkb/github.com'},
 {'date': '22-11-2021',
  'title': 'Github Library with PyTorch .',
  'url': 'mkb/github.com'}]
```

## Methods

???- note "__call__"

    Retrieve the right document using BM25.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add documents.

    **Parameters**

    - **documents**     (*list*)    
    
## References

1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)

