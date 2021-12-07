# Elastic

ElasticSearch retriever.



## Parameters

- **on** (*str*)

- **k** (*int*) – defaults to `None`

- **es** – defaults to `None`

- **index** (*str*) – defaults to `cherche`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from elasticsearch import Elasticsearch

>>> es = Elasticsearch()

>>> if es.ping():
...     retriever = retrieve.Elastic(on="title", k=2, es=es, index="test")
...
...     documents = [
...         {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
...         {"url": "mkb/github.com", "title": "Github Library with PyTorch.", "date": "22-11-2021"},
...         {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
...     ]
...
...     retriever = retriever.reset()
...     retriever = retriever.add(documents=documents)
...
...     print(retriever(q="Transformers"))
[{'date': '10-11-2021',
'title': 'Github library with PyTorch and Transformers.',
'url': 'ckb/github.com'},
{'date': '22-11-2020',
'title': 'Github Library with Pytorch and Transformers.',
'url': 'blp/github.com'}]
```

## Methods

???- note "__call__"

    ElasticSearch query.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    ElasticSearch bulk indexing.

    **Parameters**

    - **documents**     (*list*)    
    
???- note "reset"

    Delete the selected index from ElasticSearch.

    
## References

1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

