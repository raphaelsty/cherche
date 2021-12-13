# cosine

Computes cosine distance between input query embedding and documents embeddings. Higher is better.



## Parameters

- **emb_q** (*numpy.ndarray*)

    Embedding of the query.

- **emb_documents** (*list*)

    List of embeddings of the documents.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import similarity

>>> emb_q = np.array([1, 1])

>>> emb_documents = [
...     np.array([0, 10]),
...     np.array([1, 1]),
... ]

>>> print(similarity.cosine(emb_q=emb_q, emb_documents=emb_documents))
[(1, 0.9999999999999998), (0, 0.7071067811865475)]
```

