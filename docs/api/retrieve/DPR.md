# DPR

DPR as a retriever using Faiss Index.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Field to use to retrieve documents.

- **k** (*int*)

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **encoder**

- **query_encoder**

- **index** – defaults to `None`

    Index that will store the embeddings and perform the similarity search. The default index is Faiss.

- **path** (*str*) – defaults to `None`


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris", "author": "Paris"},
...    {"id": 1, "title": "Madrid", "author": "Madrid"},
...    {"id": 2, "title": "Montreal", "author": "Montreal"},
... ]

>>> retriever = retrieve.DPR(
...    key = "id",
...    on = ["title", "author"],
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    k = 2,
... )

>>> retriever.add(documents)
DPR retriever
     key: id
     on: title, author
     documents: 3

>>> print(retriever("Spain"))
[{'id': 1, 'similarity': 0.009192565994771673},
 {'id': 0, 'similarity': 0.008331424302852155}]

>>> retriever += documents

>>> print(retriever("Spain"))
[{'author': 'Madrid',
  'id': 1,
  'similarity': 0.009192565994771673,
  'title': 'Madrid'},
 {'author': 'Paris',
  'id': 0,
  'similarity': 0.008331424302852155,
  'title': 'Paris'}]
```

## Methods

???- note "__call__"

    Search for documents.

    **Parameters**

    - **q**     (*str*)    
    - **expr**     (*str*)     – defaults to `None`    
    - **consistency_level**     (*str*)     – defaults to `None`    
    - **partition_names**     (*list*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add documents to the index.

    **Parameters**

    - **documents**     (*list*)    
    - **batch_size**     (*int*)     – defaults to `64`    
    - **kwargs**    
    
