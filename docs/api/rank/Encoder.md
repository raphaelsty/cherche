# Encoder

SentenceBert Ranker.



## Parameters

- **encoder**

- **on** (*str*)

- **k** (*int*) – defaults to `None`

- **path** (*str*) – defaults to `None`

- **distance** – defaults to `<function cosine_distance at 0x7f77a873fee0>`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "title",
...    k = 2,
...    path = "encoder.pkl"
... )

>>> ranker
Encoder ranker
     on: title
     k: 2
     distance: cosine_distance
     embeddings stored at: encoder.pkl

>>> documents = [
...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
... ]

```

Pre-compute embeddings of documents
```python
>>> ranker = ranker.add(documents=documents)

>>> print(ranker(q="Transformers", documents=documents))
[{'cosine_distance': 0.6396294832229614,
  'date': '10-11-2021',
  'title': 'Github library with PyTorch and Transformers .',
  'url': 'ckb/github.com'},
 {'cosine_distance': 0.6396294832229614,
  'date': '22-11-2020',
  'title': 'Github Library with Pytorch and Transformers .',
  'url': 'blp/github.com'}]
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
    
