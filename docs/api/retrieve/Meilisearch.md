# Meilisearch

Meilisearch is a RESTful search API. It aims to be a ready-to-go solution for everyone who wants a fast and relevant search experience for their end-users.



## Parameters

- **key** (*str*)

- **on** (*str*)

    Fields to use to match the query to the documents.

- **index**

    Meilisearch index. Meilisearch will create the index if it does not exist.

- **k** (*Union[int, NoneType]*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query  will be retrieved.


## Attributes

- **type**


## Examples

```python
>>> import meilisearch
>>> from cherche import retrieve
>>> from pprint import pprint as print

>>> documents = [
...    {"key": 1, "type": "movie", "title": "Carol", "genres": ["Romance", "Drama"]},
...    {"key": 2, "type": "movie", "title": "Wonder Woman", "genres": ["Action", "Adventure"]},
...    {"key": 3, "type": "movie", "title": "Life of Pi", "genres": ["Adventure", "Drama"]}
... ]

>>> client = meilisearch.Client('http://127.0.0.1:7700', 'masterKey')

>>> retriever = retrieve.Meilisearch(
...    key="key", on=["type", "title", "genres"], k=20, index=client.index("movies"))

>>> retriever.add(documents)
Meilisearch retriever
    key: key
    on: type, title, genres
    documents: 3

>>> print(retriever("movie"))
[{'genres': ['Romance', 'Drama'],
  'key': 1,
  'similarity': 1.0,
  'title': 'Carol',
  'type': 'movie'},
 {'genres': ['Action', 'Adventure'],
  'key': 2,
  'similarity': 0.5,
  'title': 'Wonder Woman',
  'type': 'movie'},
 {'genres': ['Adventure', 'Drama'],
  'key': 3,
  'similarity': 0.3333333333333333,
  'title': 'Life of Pi',
  'type': 'movie'}]
```

## Methods

???- note "__call__"

    Retrieve the right document.

    **Parameters**

    - **q**     (*str*)    
    - **opt_params**     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Meilisearch is streaming friendly.

    **Parameters**

    - **documents**     (*list*)    
    - **batch_size**     – defaults to `128`    
    - **kwargs**    
    
???- note "reset"

## References

1. [meilisearch-python](https://github.com/meilisearch/meilisearch-python)
2. [Meilisearch documentation](https://docs.meilisearch.com/learn/getting_started/quick_start.html#setup-and-installation)
3. [Meilisearch settings](https://docs.meilisearch.com/reference/api/settings.html#settings-object)

