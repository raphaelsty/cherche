# Meilisearch

[Meilisearch](https://docs.meilisearch.com/learn/getting_started/quick_start.html) is an open-source search engine designed to meet a vast majority of needs. Requiring very little configuration to be installed, yet highly customizable.

## Run Meilisearch - Docker

Fetch the latest version of Meilisearch image from DockerHub:

```sh
docker pull getmeili/meilisearch:v0.29
```

Launch Meilisearch in development mode with a master key:

```
docker run -it --rm \
    -p 7700:7700 \
    -e MEILI_MASTER_KEY='masterKey'\
    -v $(pwd)/meili_data:/meili_data \
    getmeili/meilisearch:v0.29 \
    meilisearch --env="development"
```

We can find different ways to install Meilisearch [here](https://docs.meilisearch.com/learn/getting_started/quick_start.html#setup-and-installation).


## Meilisearch retriever

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

>>> retriever("movie")
```

```python
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