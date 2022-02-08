# Elastic

ElasticSearch retriever based on the [Python client of Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v7.15.1/).



## Parameters

- **key** (*str*)

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query  will be retrieved.

- **es** (*elasticsearch.client.Elasticsearch*) – defaults to `None`

    ElasticSearch Python client. The default configuration is used if set to None.

- **index** (*str*) – defaults to `cherche`

    Elasticsearch index to use to index documents. Elastic will create the index if it does not exist.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from elasticsearch import Elasticsearch

>>> es = Elasticsearch()

>>> if es.ping():
...
...     retriever = retrieve.Elastic(key="id", on=["title", "article"], k=2, es=es, index="test")
...
...     documents = [
...         {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...         {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...         {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
...     ]
...
...     retriever = retriever.add(documents=documents)
...     candidates = retriever(q="paris")

>>> print(candidates)
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 1.2017119,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 1.0534589,
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    ElasticSearch query.

    **Parameters**

    - **q**     (*str*)    
    - **query**     (*str*)     – defaults to `None`    
    
???- note "add"

    ElasticSearch bulk indexing.

    **Parameters**

    - **documents**     (*list*)    
    
???- note "add_embeddings"

    Store documents and embeddings inside Elasticsearch using bulk indexing. Embeddings parameter has the priority over ranker. If embeddings are provided, ElasticSearch will index documents with their embeddings. If embeddings are not provided, the Ranker will be called to compute embeddings. This method is useful if you have to deal with large corpora.

    **Parameters**

    - **documents**     (*list*)    
    - **ranker**     (*cherche.rank.base.Ranker*)     – defaults to `None`    
    - **embeddings**     (*list*)     – defaults to `None`    
    
???- note "reset"

    Delete the selected index from ElasticSearch.

    
## References

1. [Python Elasticsearch Client](https://elasticsearch-py.readthedocs.io/en/v7.15.1/)

