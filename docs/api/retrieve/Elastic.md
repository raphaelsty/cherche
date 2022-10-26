# Elastic

ElasticSearch retriever based on the [Python client of Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v7.15.1/).



## Parameters

- **key** (*str*)

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query  will be retrieved.

- **es** (*elasticsearch.Elasticsearch*) – defaults to `None`

    ElasticSearch Python client. The default configuration is used if set to None.

- **index** (*str*) – defaults to `cherche`

    Elasticsearch index to use to index documents. Elastic will create the index if it does not exist.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from elasticsearch import Elasticsearch
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

>>> es = Elasticsearch(hosts="http://localhost:9200")

>>> if es.ping():
...    retriever = retrieve.Elastic(key="id", on=["title", "author"], k=2, es=es, index="test")
...
...    documents = [
...         {"id": 0, "title": "Paris", "author": "Wiki"},
...         {"id": 1, "title": "Eiffel tower", "author": "Wiki"},
...         {"id": 2, "title": "Montreal", "author": "Wiki"},
...    ]
...
...    retriever = retriever.add(documents=documents)
...    candidates = retriever(q="paris")
```

## Methods

???- note "__call__"

    ElasticSearch query.

    **Parameters**

    - **q**     (*str*)    
    - **query**     (*str*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    ElasticSearch bulk indexing.

    **Parameters**

    - **documents**     (*list*)    
    - **batch_size**     – defaults to `128`    
    - **kwargs**    
    
???- note "reset"

    Delete the selected index from ElasticSearch.

    
## References

1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

