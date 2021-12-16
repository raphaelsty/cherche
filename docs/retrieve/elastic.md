# Elastic

The Elastic module is a wrapper for the [Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v8.0.0a1/) Python client. This module allows you to use your own Elasticsearch server to integrate it into a neural search pipeline.

To use Elasticsearch as a retriever you will need to install Elasticsearch and start the server, information is available [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).

You can find all the information associated with the `es` parameter which establishes the connection to your Elasticsearch session on the [official documentation](https://elasticsearch-py.readthedocs.io/en/v8.0.0a1/api.html#module-elasticsearch).

Once your Elasticsearch server is up and running, you can connect to it with Search to index documents and for queries.

```python
>>> from cherche import retrieve
>>> from elasticsearch import Elasticsearch

# Elasticsearch client
>>> es = Elasticsearch(hosts="localhost:9200")

# Ask to cherche on title and article, retrieves the top 30 results, uses and create the index cherche if it does not exist.
>>> retriever = retrieve.Elastic(on=["title", "article"], k=30, es=es, index="cherche")

>>> documents = [
...    {
...        "article": "Paris is the capital and most populous city of France",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    },
...    {
...        "article": "The City of Paris is the centre and seat of government of the region and province of Île-de-France .",
...        "title": "Paris",
...        "url": "https://en.wikipedia.org/wiki/Paris"
...    }
... ]

>>> retriever.add(documents=documents)

>>> retriever(q="science")
```

```python
[{"article": "Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.",
  "title": "Paris",
  "url": "https://en.wikipedia.org/wiki/Paris"}]
```

Warning, you can empty the index used by Cherche using:

```python
retriever.reset()
```
