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

>>> retriever = retrieve.BM25Okapi(on="article", k=3, k1=1.5, b=0.75, epsilon=0.25)

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retriever.add(documents=documents)

>>> retriever
BM25Okapi retriever
    on: article
    documents: 3

>>> print(retriever(q="France"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'}]

>>> retriever.add(documents=documents)
BM25Okapi retriever
    on: article
    documents: 6
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
