# dot_similarity

Computes dot product between input query embedding and documents embeddings. Higher is better.



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

>>> print(distance.dot_similarity(emb_q=emb_q, emb_documents=emb_documents))
[(0, 10), (1, 2)]
```

