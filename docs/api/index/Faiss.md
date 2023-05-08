# Faiss

Faiss index dedicated to vector search.



## Parameters

- **key**

    Identifier field for each document.

- **index** – defaults to `None`

    Faiss index to use.

- **normalize** (*bool*) – defaults to `True`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import index
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris France"},
...    {"id": 1, "title": "Madrid Spain"},
...    {"id": 2, "title": "Montreal Canada"}
... ]

>>> encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

>>> faiss_index = index.Faiss(key="id")
>>> faiss_index = faiss_index.add(
...    documents = documents,
...    embeddings = encoder.encode([document["title"] for document in documents]),
... )

>>> print(faiss_index(embeddings=encoder.encode(["Spain", "Montreal"])))
[[{'id': 1, 'similarity': 0.6544566197822951},
  {'id': 0, 'similarity': 0.5405466290777285},
  {'id': 2, 'similarity': 0.48717489472604614}],
 [{'id': 2, 'similarity': 0.7372165680578416},
  {'id': 0, 'similarity': 0.5185646665953703},
  {'id': 1, 'similarity': 0.4834444940712032}]]

>>> documents = [
...    {"id": 3, "title": "Paris France"},
...    {"id": 4, "title": "Madrid Spain"},
...    {"id": 5, "title": "Montreal Canada"}
... ]

>>> faiss_index = faiss_index.add(
...    documents = documents,
...    embeddings = encoder.encode([document["title"] for document in documents]),
... )

>>> print(faiss_index(embeddings=encoder.encode(["Spain", "Montreal"]), k=4))
[[{'id': 1, 'similarity': 0.6544566197822951},
  {'id': 4, 'similarity': 0.6544566197822951},
  {'id': 0, 'similarity': 0.5405466290777285},
  {'id': 3, 'similarity': 0.5405466290777285}],
 [{'id': 2, 'similarity': 0.7372165680578416},
  {'id': 5, 'similarity': 0.7372165680578416},
  {'id': 0, 'similarity': 0.5185646665953703},
  {'id': 3, 'similarity': 0.5185646665953703}]]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **embeddings**     (*numpy.ndarray*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    
???- note "add"

    Add documents to the faiss index and export embeddings if the path is provided. Streaming friendly.

    **Parameters**

    - **documents**     (*list*)    
    - **embeddings**     (*numpy.ndarray*)    
    
## References

1. [Faiss](https://github.com/facebookresearch/faiss)

