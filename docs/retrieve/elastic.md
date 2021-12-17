# Elastic

The Elastic module is a wrapper for the [Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v8.0.0a1/) Python client. This module allows you to use your own Elasticsearch server to integrate it into a neural search pipeline.

To use Elasticsearch as a retriever you will need to install Elasticsearch and start the server, information is available [here](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html).

You can find all the information associated with the `es` parameter which establishes the connection to your Elasticsearch session on the [official documentation](https://elasticsearch-py.readthedocs.io/en/v8.0.0a1/api.html#module-elasticsearch).

Once your Elasticsearch server is up and running, you can connect to it with Search to index documents and for queries.

The `retrieve.Elastic` retriever is the right solution if you want to implement a neural search pipeline on a large corpus, the Elastic retriever is the right solution. The retriever allows the documents and pre-computed embeddings of the ranker to be indexed on Elasticsearch so as not to overload the RAM.

## Elastic retriever

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

## Write your own Elastic query

When using the Elastic retriever, we can customise the Elasticsearch query to fit our needs.

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

>>> q = "science"

>>> query = {
...    "query": {
...        "multi_match": {
...            "query": q,
...            "type": "most_fields",
...            "fields": ["title", "article"],
...            "operator": "or",
...        }
...    },
... }

>>> retriever.add(documents=documents)

>>> retriever(**{"q": q, "query": query})
```

```python
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris'}]
```

## Upload documents embeddings on Elastic

If we are working with millions of documents, we will need to store the documents and embeddings on disk rather than in RAM. To do this, we can declare an Elastic retriever and index the documents and embeddings in the ranker. When a query is made, Elasticsearch will propose the most relevant candidates and the ranker will retrieve the embeddings from Elasticsearch and finally re-rank the documents. No documents or embeddings will be stored directly in the models.

It is essential to have a GPU when pre-calculating the embeddings when you want to search through millions of documents.

```python
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer
>>> from elasticsearch import Elasticsearch

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

# Elasticsearch client, in localhost here but it could be remotly.
>>> es = Elasticsearch(hosts="localhost:9200")

>>> retriever = retrieve.Elastic(on=["title", "article"], k=30, es=es, index="cherche")

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = ["title", "article"],
...    k = 10,
... )

# Store embeddings and documents in Elasticsearch
>>> retriever.add_embeddings(documents=documents, ranker=ranker)

>>> search = retriever + ranker

>>> search("science")
```

```python
[{'article': 'Paris has been one of Europe major centres of finance, diplomacy , commerce , fashion , gastronomy , science , and arts.',
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'similarity': 0.0890004881677473}]
```
