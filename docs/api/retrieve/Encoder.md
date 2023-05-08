# Encoder

Encoder as a retriever using Faiss Index.



## Parameters

- **encoder**

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Field to use to retrieve documents.

- **normalize** (*bool*) – defaults to `True`

    Whether to normalize the embeddings before adding them to the index in order to measure cosine similarity.

- **k** (*Optional[int]*) – defaults to `None`

- **batch_size** (*int*) – defaults to `64`

- **index** – defaults to `None`

    Faiss index that will store the embeddings and perform the similarity search.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris France"},
...    {"id": 1, "title": "Madrid Spain"},
...    {"id": 2, "title": "Montreal Canada"}
... ]

>>> retriever = retrieve.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title"],
... )

>>> retriever.add(documents, batch_size=1)
Encoder retriever
    key      : id
    on       : title
    documents: 3

>>> print(retriever("Spain", k=2))
[{'id': 1, 'similarity': 0.6544566453117681},
 {'id': 0, 'similarity': 0.5405465419981407}]

>>> print(retriever(["Spain", "Montreal"], k=2))
[[{'id': 1, 'similarity': 0.6544566453117681},
  {'id': 0, 'similarity': 0.54054659424589}],
 [{'id': 2, 'similarity': 0.7372165680578416},
  {'id': 0, 'similarity': 0.5185645704259234}]]
```

## Methods

???- note "__call__"

    Retrieve documents from the index.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add documents to the index.

    **Parameters**

    - **documents**     (*List[Dict[str, str]]*)    
    - **batch_size**     (*int*)     – defaults to `64`    
    - **kwargs**    
    
