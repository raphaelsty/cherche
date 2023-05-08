# TopK

Filter top k documents in pipeline.



## Parameters

- **k** (*int*)

    Number of documents to keep.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, rank, utils
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris France"},
...    {"id": 1, "title": "Madrid Spain"},
...    {"id": 2, "title": "Montreal Canada"}
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents)

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title"],
... )

>>> pipeline = retriever + ranker + utils.TopK(k=2)
>>> pipeline.add(documents=documents)
TfIdf retriever
    key      : id
    on       : title, article
    documents: 3
Encoder ranker
    key       : id
    on        : title
    normalize : True
    embeddings: 3
Filter TopK
    k: 2

>>> print(pipeline(q="Paris Madrid Montreal", k=2))
[{'id': 0, 'similarity': 0.62922895}, {'id': 2, 'similarity': 0.61419094}]
```

## Methods

???- note "__call__"

    Filter top k documents in pipeline.

    **Parameters**

    - **documents**     (*List[List[Dict[str, str]]]*)    
    - **kwargs**    
    
