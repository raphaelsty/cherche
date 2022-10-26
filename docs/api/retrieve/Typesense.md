# Typesense

Typesense retriever.



## Parameters

- **key** (*str*)

- **on** (*Union[str, list]*)

- **collection**

- **k** (*Union[int, NoneType]*) – defaults to `None`


## Attributes

- **type**


## Examples

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

>>> exist = False
>>> for collection in client.collections.retrieve():
...     if collection["name"] == "documentation":
...         exist = True

>>> if not exist:
...     response = client.collections.create({
...         "name": "documentation",
...         "fields": [
...             {"name": "id", "type": "string"},
...             {"name": "title", "type": "string"},
...             {"name": "author", "type": "string", "optional": True},
...         ],
...     })

>>> retriever = retrieve.Typesense(
...     key="id",
...     on=["title", "author"],
...     collection=client.collections['documentation']
... )

>>> retriever.add(documents)
Typesense retriever
    key: id
    on: title, author
    documents: 3

>>> retriever("madrid paris")
[{'author': 'Madrid', 'title': 'Madrid', 'similarity': 1.0, 'id': 1}]
```

## Methods

???- note "__call__"

    Search for documents.

    **Parameters**

    - **q**     (*str*)    
    - **query**     (*Union[dict, NoneType]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

## References

1. [Typesense Github](https://github.com/typesense/typesense)
2. [Documentation](https://typesense.org/docs/0.23.1/api/)

