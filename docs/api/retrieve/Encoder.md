# Encoder

Encoder as a retriever using Faiss Index.



## Parameters

- **encoder**

- **on** (*Union[str, list]*)

    Field to use to retrieve documents.

- **k** (*int*)

    Number of documents to retrieve.

- **path** (*str*) – defaults to `None`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = ["title", "article"],
...    k = 2,
...    path = "retriever_encoder.pkl"
... )

>>> retriever.add(documents)
Encoder retriever
     on: title, article
     documents: 3

>>> print(retriever("Paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 1.472814254853544,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 1.0293491728070765,
  'title': 'Eiffel tower'}]

>>> retriever.add(documents)
Encoder retriever
     on: title, article
     documents: 6

>>> print(retriever("Paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 1.472814254853544,
  'title': 'Paris'},
 {'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 1.472814254853544,
  'title': 'Paris'}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add documents to the faiss index and export embeddings if the path is provided.

    **Parameters**

    - **documents**     (*list*)    
    
???- note "build_faiss"

    Build faiss index.

    **Parameters**

    - **tree**     (*faiss.swigfaiss.IndexFlatL2*)    
    - **documents_embeddings**     (*list*)    
    
???- note "dump_embeddings"

    Dump embeddings to the selected directory.

    **Parameters**

    - **embeddings**     (*dict*)    
    - **path**     (*str*)    
    
???- note "load_embeddings"

    Load embeddings from an existing directory.

    - **path**     (*str*)    
    
## References

1. [Faiss](https://github.com/facebookresearch/faiss)

