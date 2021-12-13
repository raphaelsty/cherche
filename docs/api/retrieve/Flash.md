# Flash

FlashText Retriever.



## Parameters

- **on** (*str*)

    Field to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **keywords** – defaults to `None`

    Keywords extractor from [FlashText](https://github.com/vi3k6i5/flashtext). If set to None, a default one is created.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.Flash(on="tag", k=2)

>>> documents = [
...     {"url": "ckb/github.com", "tag": "Transformers", "date": "10-11-2021", "label": "Transformers are heavy."},
...     {"url": "mkb/github.com", "tag": ["Transformers", "Pytorch"], "date": "22-11-2021", "label": "Transformers with Pytorch"},
...     {"url": "blp/github.com", "tag": "Github", "date": "22-11-2020", "label": "Github is a great tool."},
... ]

>>> retriever = retriever.add(documents=documents)

>>> print(retriever(q="Transformers with Pytorch"))
[{'date': '10-11-2021',
  'label': 'Transformers are heavy.',
  'tag': 'Transformers',
  'url': 'ckb/github.com'},
 {'date': '22-11-2021',
  'label': 'Transformers with Pytorch',
  'tag': ['Transformers', 'Pytorch'],
  'url': 'mkb/github.com'}]
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

