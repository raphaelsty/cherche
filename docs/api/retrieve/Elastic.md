# Elastic

ElasticSearch retriever.



## Parameters

- **on** (*str*)

    Field to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **es** – defaults to `None`

    ElasticSearch Python client. The default configuration is used if set to None.

- **index** (*str*) – defaults to `cherche`

    Elasticsearch index to use to index documents. Elastic will create the index if it does not exist.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from elasticsearch import Elasticsearch

>>> es = Elasticsearch()

>>> if es.ping():
...
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
...     candidates = retriever(q="Transformers")
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

