# BM25Plus

BM25Plus



## Parameters

- **on** (*str*)

- **tokenizer** – defaults to `None`

- **k** (*int*) – defaults to `None`

- **k1** – defaults to `1.5`

- **b** – defaults to `0.75`

- **delta** – defaults to `0.5`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.BM25Plus(on="title", k=3, k1=1.5, b=0.75, delta=0.5)

>>> documents = [
...     {"url": "ckb/github.com", "title": "It is quite windy in London.", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch and Transformers .", "date": "22-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with Transformers .", "date": "22-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch and Transformers .", "date": "22-11-2021"},
... ]

>>> retriever = retriever.add(documents=documents)

>>> retriever
BM25Plus retriever
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
2. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)

