# BM25L

BM25L model from [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25).



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **tokenizer** – defaults to `None`

    Tokenizer to use, the default one split on spaces. This tokenizer should have a `tokenizer.__call__` method that returns the list of tokenized tokens.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **k1** (*float*) – defaults to `1.5`

    Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].

- **b** (*float*) – defaults to `0.75`

    Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].

- **delta** (*float*) – defaults to `0.5`

    Smoothing parameter defined in [Improvements to BM25 and Language Models Examined[http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf].



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.BM25L(on=["title", "article"], k=3, k1=1.5, b=0.75, delta=0.5)

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retriever.add(documents=documents)

>>> retriever
BM25L retriever
    on: title, article
    documents: 3

>>> print(retriever(q="Paris"))
[{'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'title': 'Eiffel tower'},
 {'article': 'This town is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'}]

>>> retriever.add(documents=documents)
BM25L retriever
    on: title, article
    documents: 6
```

## Methods

???- note "__call__"

    Retrieve the right document using BM25.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add documents to the retriever.

    **Parameters**

    - **documents**     (*list*)    
    
## References

1. [Rank-BM25: A two line search engine](https://github.com/dorianbrown/rank_bm25)
2. [Improvements to BM25 and Language Models Examined](http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf)
