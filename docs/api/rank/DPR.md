# DPR

DPR is dedicated to rank documents using distinct models to encode the query and the documents contents.



## Parameters

- **encoder**

- **query_encoder**

- **on** (*str*)

- **k** (*int*) – defaults to `None`

- **path** (*str*) – defaults to `None`

- **metric** – defaults to `<function dot_similarity at 0x7fdfb2f7f0d0>`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> ranker = rank.DPR(
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    on = "title",
...    k = 2,
...    path = "dpr.pkl"
... )

>>> ranker
DPR ranker
     on: title
     k: 2
     Metric: dot_similarity
     Embeddings stored at: dpr.pkl

>>> documents = [
...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
... ]

```

Pre-compute embeddings of documents
```python
>>> ranker = ranker.add(documents=documents)

>>> print(ranker(q="Transformers", documents=documents, k=2))
[{'date': '10-11-2021',
  'dot_similarity': 54.095573,
  'title': 'Github library with PyTorch and Transformers .',
  'url': 'ckb/github.com'},
 {'date': '22-11-2020',
  'dot_similarity': 54.095573,
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
    
