# Encoder

SentenceBert Ranker.



## Parameters

- **encoder**

- **on** (*str*)

- **k** (*int*) – defaults to `None`

- **path** (*str*) – defaults to `None`

- **similarity** – defaults to `<function cosine at 0x7fb6020d3af0>`



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

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "article",
...    k = 2,
...    path = "encoder.pkl"
... )

>>> ranker.add(documents=documents)
Encoder ranker
     on: article
     k: 2
     similarity: cosine
     embeddings stored at: encoder.pkl

>>> print(ranker(q="Paris", documents=documents))
[{'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 0.49121392,
  'title': 'Eiffel tower'},
 {'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 0.44376045,
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
    
