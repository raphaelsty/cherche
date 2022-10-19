# Typesense

[Typesense](https://typesense.org/docs/guide/install-typesense.html#option-1-typesense-cloud) is an open-source, typo-tolerant search engine. TypeSense

## Typesense - Docker

We can launch Typesense via docker. The documents indexed via the docker instance are persistent.

We can fetch the latest version of Typesense image from DockerHub:

```sh
docker pull typesense/typesense:0.23.1
```

Then we can launch Typesense with a master key:

```sh
docker run -p 8108:8108 -v/tmp/data:/data typesense/typesense:0.23.1 --data-dir /data --api-key=Hu52dwsas2AdxdE
```

There are differents ways to install Typesense, informations are available via the [documentation](https://typesense.org/docs/guide/install-typesense.html#option-1-typesense-cloud).

## Typesense retriever

Before creating a Typesense retriever, creating a collection with the list of fields and their associated properties is necessary. Informations to create a collection are available on the Typesense [documentation](https://typesense.org/docs/0.23.1/api/collections.html#create-a-collection).


Creation of the collection `documentation`:


```python
>>> import typesense

>>> documents = [
...    {"id": 0, "title": "Paris", "author": "Paris"},
...    {"id": 1, "title": "Madrid", "author": "Madrid"},
...    {"id": 2, "title": "Montreal", "author": "Montreal"},
... ]

>>> client = typesense.Client({
...    'api_key': 'Hu52dwsas2AdxdE',
...    'nodes': [{
...        'host': 'localhost',
...        'port': '8108',
...        'protocol': 'http'
...    }],
...    'connection_timeout_seconds': 2
... })

>>> response = client.collections.create({
...     "name": "documentation",
...     "fields": [
...         {"name": "id", "type": "string"},
...         {"name": "title", "type": "string"},
...         {"name": "author", "type": "string", "optional": True},
...     ],
... })
```

Initialization of our retriever:

```python
>>> from cherche import retrieve
>>> import typesense

>>> documents = [
...    {"id": 0, "title": "Paris", "author": "Paris"},
...    {"id": 1, "title": "Madrid", "author": "Madrid"},
...    {"id": 2, "title": "Montreal", "author": "Montreal"},
... ]

>>> client = typesense.Client({
...    'api_key': 'Hu52dwsas2AdxdE',
...    'nodes': [{
...        'host': 'localhost',
...        'port': '8108',
...        'protocol': 'http'
...    }],
...    'connection_timeout_seconds': 2
... })

>>> retriever = retrieve.Typesense(
...     key="id",
...     on=["title", "author"],
...     collection=client.collections['documentation']
... )

>>> retriever.add(documents)

>>> retriever("madrid paris")
```

```python
[{'author': 'Madrid', 'title': 'Madrid', 'similarity': 1.0, 'id': 1}]
```