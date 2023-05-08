# Embedding

Collaborative filtering as a ranker. Recommend is compatible with the library [Implicit](https://github.com/benfred/implicit).



## Parameters

- **key** (*'str'*)

    Field identifier of each document.

- **normalize** (*'bool'*) – defaults to `True`

    If set to True, the similarity measure is cosine similarity, if set to False, similarity measure is dot product.

- **k** (*'typing.Optional[int]'*) – defaults to `None`

- **batch_size** (*'int'*) – defaults to `1024`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": "a", "title": "Paris"},
...    {"id": "b", "title": "Madrid"},
...    {"id": "c", "title": "Montreal"},
... ]

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
>>> embeddings_documents = encoder.encode([
...    document["title"] for document in documents
... ])

>>> recommend = rank.Embedding(
...    key="id",
... )

>>> recommend.add(
...    documents=documents,
...    embeddings_documents=embeddings_documents,
... )
Embedding ranker
    key      : id
    documents: 3
    normalize: True

>>> match = recommend(
...     q=encoder.encode("Paris"),
...     documents=documents,
...     k=2
... )

>>> print(match)
[{'id': 'a', 'similarity': 1.0, 'title': 'Paris'},
 {'id': 'c', 'similarity': 0.57165134, 'title': 'Montreal'}]

>>> queries = [
...    "Paris",
...    "Madrid",
...    "Montreal"
... ]

>>> match = recommend(
...     q=encoder.encode(queries),
...     documents=[documents] * 3,
...     k=2
... )

>>> print(match)
[[{'id': 'a', 'similarity': 1.0, 'title': 'Paris'},
  {'id': 'c', 'similarity': 0.57165134, 'title': 'Montreal'}],
 [{'id': 'b', 'similarity': 1.0, 'title': 'Madrid'},
  {'id': 'a', 'similarity': 0.49815434, 'title': 'Paris'}],
 [{'id': 'c', 'similarity': 0.9999999, 'title': 'Montreal'},
  {'id': 'a', 'similarity': 0.5716514, 'title': 'Paris'}]]
```

## Methods

???- note "__call__"

    Retrieve documents from user id.

    **Parameters**

    - **q**     (*'np.ndarray'*)    
    - **documents**     (*'typing.Union[typing.List[typing.List[typing.Dict[str, str]]], typing.List[typing.Dict[str, str]]]'*)    
    - **k**     (*'typing.Optional[int]'*)     – defaults to `None`    
    - **batch_size**     (*'typing.Optional[int]'*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add embeddings both documents and users.

    **Parameters**

    - **documents**     (*'list'*)    
    - **embeddings_documents**     (*'typing.List[np.ndarray]'*)    
    - **kwargs**    
    
???- note "encode_rank"

    Encode documents and rank them according to the query.

    **Parameters**

    - **embeddings_queries**     (*numpy.ndarray*)    
    - **documents**     (*List[List[Dict[str, str]]]*)    
    - **k**     (*int*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    
???- note "rank"

    Rank inputs documents ordered by relevance among the top k.

    **Parameters**

    - **embeddings_documents**     (*Dict[str, numpy.ndarray]*)    
    - **embeddings_queries**     (*numpy.ndarray*)    
    - **documents**     (*List[List[Dict[str, str]]]*)    
    - **k**     (*int*)    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    
