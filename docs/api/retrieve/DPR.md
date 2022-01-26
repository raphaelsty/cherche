# DPR

DPR as a retriever using Faiss Index.



## Parameters

- **encoder**

- **query_encoder**

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Field to use to retrieve documents.

- **k** (*int*)

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **path** (*str*) – defaults to `None`


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
...    path = "retriever_dpr.pkl"
... )

>>> retriever.add(documents)
DPR retriever
     key: id
     on: title, article
     documents: 3

>>> print(retriever("Paris"))
[{'id': 0, 'similarity': 0.011120470176519816},
 {'id': 2, 'similarity': 0.010158280600646162}]

>>> documents = [
...    {"id": 3, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 4, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 5, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever.add(documents)
DPR retriever
     key: id
     on: title, article
     documents: 6

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
...    {"id": 3, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 4, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 5, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever += documents

>>> print(retriever("Paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 3,
  'similarity': 0.011120470176519816,
  'title': 'Paris'},
 {'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.011120470176519816,
  'title': 'Paris'}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add documents to the faiss index and export embeddings if the path is provided. Streaming friendly.

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
