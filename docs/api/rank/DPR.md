# DPR

DPR is dedicated to rank documents using distinct models to encode the query and the documents contents.



## Parameters

- **encoder**

- **query_encoder**

- **on** (*str*)

- **k** (*int*) – defaults to `None`

- **path** (*str*) – defaults to `None`

- **similarity** – defaults to `<function dot at 0x7fb6020dc3a0>`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    on = "article",
...    k = 2,
...    path = "test_dpr.pkl"
... )

>>> ranker.add(documents=documents)
DPR ranker
     on: article
     k: 2
     similarity: dot
     embeddings stored at: test_dpr.pkl

>>> print(ranker(q="Paris", documents=documents, k=2))
[{'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 69.8168,
  'title': 'Eiffel tower'},
 {'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 67.30965,
  'title': 'Paris'}]
```

## Methods

???- note "__call__"

    Encode inputs query and ranks documents based on the similarity between the query and the selected field of the documents.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
???- note "add"

    Pre-compute embeddings and store them at the selected path.

    **Parameters**

    - **documents**    
    
???- note "dump_embeddings"

    Dump embeddings to the selected directory.

    **Parameters**

    - **embeddings**    
    - **path**     (*str*)    
    
???- note "load_embeddings"

    Load embeddings from an existing directory.

    - **path**     (*str*)    
    
