# Encoder

Encoder as a retriever using Faiss Index.



## Parameters

- **encoder**

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Field to use to retrieve documents.

- **k** (*int*)

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

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

>>> retriever = retrieve.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title", "author"],
...    k = 2,
... )

>>> retriever.add(documents)
Encoder retriever
     key: id
     on: title, author
     documents: 3

>>> print(retriever("Spain"))
[{'id': 1, 'similarity': 1.1885032405192992},
 {'id': 0, 'similarity': 0.8492543139964137}]
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
    
## References

1. [Faiss](https://github.com/facebookresearch/faiss)

