# Embedding

The Embedding retriever is dedicated to perform IR on embeddings calculated by the user rather than Cherche.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **index** – defaults to `None`

    Faiss index that will store the embeddings and perform the similarity search.

- **normalize** (*bool*) – defaults to `True`

    Whether to normalize the embeddings before adding them to the index in order to measure cosine similarity.

- **k** (*Optional[int]*) – defaults to `None`

- **batch_size** (*int*) – defaults to `1024`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> recommend = retrieve.Embedding(
...    key="id",
... )

>>> documents = [
...    {"id": "a", "title": "Paris", "author": "Paris"},
...    {"id": "b", "title": "Madrid", "author": "Madrid"},
...    {"id": "c", "title": "Montreal", "author": "Montreal"},
... ]

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
>>> embeddings_documents = encoder.encode([
...    document["title"] for document in documents
... ])

>>> recommend.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
... )
Embedding retriever
    key      : id
    documents: 3

>>> queries = [
...    "Paris",
...    "Madrid",
...    "Montreal"
... ]

>>> embeddings_queries = encoder.encode(queries)
>>> print(recommend(embeddings_queries, k=2))
[[{'id': 'a', 'similarity': 1.0},
  {'id': 'c', 'similarity': 0.5385907831761005}],
 [{'id': 'b', 'similarity': 1.0},
  {'id': 'a', 'similarity': 0.4990788711758875}],
 [{'id': 'c', 'similarity': 1.0},
  {'id': 'a', 'similarity': 0.5385907831761005}]]

>>> embeddings_queries = encoder.encode("Paris")
>>> print(recommend(embeddings_queries, k=2))
[{'id': 'a', 'similarity': 0.9999999999989104},
 {'id': 'c', 'similarity': 0.5385907485958683}]
```

## Methods

???- note "__call__"

    Retrieve documents from the index.

    **Parameters**

    - **q**     (*numpy.ndarray*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add embeddings both documents and users.

    **Parameters**

    - **documents**     (*list*)    
    - **embeddings_documents**     (*numpy.ndarray*)    
    - **kwargs**    
    
