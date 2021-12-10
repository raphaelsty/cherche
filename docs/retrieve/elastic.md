# Elastic

The Elastic module is a wrapper for the [Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v8.0.0a1/) Python client. This module allows you to use your own Elasticsearch server to integrate it into a neural search pipeline.

To use Elasticsearch as a retriever you will need to install Elasticsearch and start the server, information is available [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).

You can find all the information associated with the `es` parameter which establishes the connection to your Elasticsearch session on the [official] documentation [https://elasticsearch-py.readthedocs.io/en/v8.0.0a1/api.html#module-elasticsearch].

Once your Elasticsearch server is up and running, you can connect to it with Search to index documents and for queries.

```python
>>> from cherche import retrieve
>>> from elasticsearch import Elasticsearch

# Elasticsearch client
>>> es = Elasticsearch()

# Ask to cherche on article, retrieves the top 30 results, uses and create the index cherche if it does not exist.
>>> retriever = retrieve.Elastic(on="article", k=30, es=es, index="cherche")

>>> documents = [
...         {"url": "ckb/github.com", "article": "Github library with PyTorch and Transformers.", "date": "10-11-2021"},
...         {"url": "mkb/github.com", "article": "Github Library with PyTorch.", "date": "22-11-2021"},
...         {"url": "blp/github.com", "article": "Github Library with Pytorch and Transformers.", "date": "22-11-2020"},
... ]

# Add the documents to the index
>>> retriever = retriever.add(documents=documents)

# Retrieve documents
>>> retriever(q="Transformers")
[{'date': '10-11-2021',
'title': 'Github library with PyTorch and Transformers.',
'url': 'ckb/github.com'},
{'date': '22-11-2020',
'title': 'Github Library with Pytorch and Transformers.',
'url': 'blp/github.com'}]
```

Warning: You can empty the index used by Cherche using:

```python
retriever.reset()
```
