# cosine_distance

Computes cosine distance between input query embedding and documents embeddings. Lower is better.



## Parameters

- **emb_q** (*numpy.ndarray*)

- **emb_documents** (*list*)



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import distance

>>> emb_q = np.array([1, 1])

>>> emb_documents = [
...     np.array([0, 10]),
...     np.array([1, 1]),
... ]

>>> print(distance.cosine_distance(emb_q=emb_q, emb_documents=emb_documents))
[(1, 0.0), (0, 0.29289321881345254)]
```

