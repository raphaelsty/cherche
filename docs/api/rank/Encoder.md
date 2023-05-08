# Encoder

Sentence Transformer as a ranker. This ranker is compatible with any SentenceTransformer.



## Parameters

- **on** (*Union[str, List[str]]*)

    Fields on wich encoder will perform similarity matching.

- **key** (*str*)

    Field identifier of each document.

- **encoder**

    Encoding function dedicated to both documents and queries.

- **normalize** (*bool*) – defaults to `True`

    If set to True, the similarity measure is cosine similarity, if set to False, similarity measure is dot product.

- **k** (*Optional[int]*) – defaults to `None`

- **batch_size** (*int*) – defaults to `64`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris France"},
...    {"id": 1, "title": "Madrid Spain"},
...    {"id": 2, "title": "Montreal Canada"}
... ]

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title"],
... )

>>> ranker.add(documents=documents)
Encoder ranker
    key       : id
    on        : title
    normalize : True
    embeddings: 3

>>> match = ranker(
...     q="Paris",
...     documents=documents
... )

>>> print(match)
[{'id': 0, 'similarity': 0.7127624, 'title': 'Paris France'},
 {'id': 1, 'similarity': 0.5497405, 'title': 'Madrid Spain'},
 {'id': 2, 'similarity': 0.50252455, 'title': 'Montreal Canada'}]

>>> match = ranker(
...     q=["Paris France", "Madrid Spain"],
...     documents=[documents + [{"id": 3, "title": "Paris"}]] * 2,
...     k=2,
... )

>>> print(match)
[[{'id': 0, 'similarity': 0.99999994, 'title': 'Paris France'},
  {'id': 1, 'similarity': 0.856435, 'title': 'Madrid Spain'}],
 [{'id': 1, 'similarity': 1.0, 'title': 'Madrid Spain'},
  {'id': 0, 'similarity': 0.856435, 'title': 'Paris France'}]]
```

## Methods

???- note "__call__"

    Encode input query and ranks documents based on the similarity between the query and the selected field of the documents.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **documents**     (*Union[List[List[Dict[str, str]]], List[Dict[str, str]]]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Pre-compute embeddings and store them at the selected path.

    **Parameters**

    - **documents**     (*List[Dict[str, str]]*)    
    - **batch_size**     (*int*)     – defaults to `64`    
    
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
    
