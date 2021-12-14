# Elastic

ElasticSearch retriever based on the [Python client of Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v7.15.1/).



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **es** (*elasticsearch.client.Elasticsearch*) – defaults to `None`

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
...     retriever = retrieve.Elastic(on=["title", "article"], k=2, es=es, index="test")
...
...     documents = [
...         {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...         {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...         {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
...     ]
...
...     retriever = retriever.reset()
...     retriever = retriever.add(documents=documents)
...     candidates = retriever(q="paris")
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

