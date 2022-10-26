# Faiss

Faiss index dedicated to vector search.



## Parameters

- **key**

    Identifier field for each document.

- **index** – defaults to `None`

    Faiss index to use.



## Examples

```python
>>> from cherche import index
>>> from sentence_transformers import SentenceTransformer
>>> from pprint import pprint as print

>>> documents = [
...    {"id": 0, "title": "Paris"},
...    {"id": 1, "title": "Madrid"},
...    {"id": 2, "title": "Paris"}
... ]

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

>>> faiss_index = index.Faiss(key="id")
>>> faiss_index = faiss_index.add(
...    documents = documents,
...    embeddings = encoder.encode([document["title"] for document in documents]),
... )

>>> print(faiss_index(embedding = encoder.encode(["Spain"])))
[{'id': 1, 'similarity': 1.5076334135501044},
 {'id': 2, 'similarity': 0.9021741164485997},
 {'id': 0, 'similarity': 0.9021741164485997}]

>>> documents = [
...    {"id": 3, "title": "Paris"},
...    {"id": 4, "title": "Madrid"},
...    {"id": 5, "title": "Paris"}
... ]

>>> faiss_index = faiss_index.add(
...    documents = documents,
...    embeddings = encoder.encode([document["title"] for document in documents]),
... )

>>> print(faiss_index(embedding = encoder.encode(["Spain"]), k=4))
[{'id': 1, 'similarity': 1.5076334135501044},
 {'id': 4, 'similarity': 1.5076334135501044},
 {'id': 2, 'similarity': 0.9021741164485997},
 {'id': 3, 'similarity': 0.9021741164485997}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **embedding**     (*numpy.ndarray*)    
    - **k**     (*int*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add documents to the faiss index and export embeddings if the path is provided. Streaming friendly.

    **Parameters**

    - **documents**     (*list*)    
    - **embeddings**     (*numpy.ndarray*)    
    
## References

1. [Faiss](https://github.com/facebookresearch/faiss)

