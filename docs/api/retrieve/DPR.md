# DPR

DPR as a retriever using Faiss Index.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Field to use to retrieve documents.

- **encoder**

- **query_encoder**

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

>>> retriever = retrieve.DPR(
...    key = "id",
...    on = ["title"],
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    normalize = True,
... )

>>> retriever.add(documents)
DPR retriever
    key      : id
    on       : title
    documents: 3

>>> print(retriever("Spain", k=2))
[{'id': 1, 'similarity': 0.5534179127892946},
 {'id': 0, 'similarity': 0.48604427456660426}]

>>> print(retriever(["Spain", "Montreal"], k=2))
[[{'id': 1, 'similarity': 0.5534179492996913},
  {'id': 0, 'similarity': 0.4860442182428353}],
 [{'id': 2, 'similarity': 0.5451990410703741},
  {'id': 0, 'similarity': 0.47405722260691213}]]
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
    
