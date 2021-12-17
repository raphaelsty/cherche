# DPR

DPR is dedicated to rank documents using distinct models to encode the query and the documents contents.



## Parameters

- **encoder**

    Encoding function dedicated to documents.

- **query_encoder**

    Encoding function dedicated to the query.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **path** (*str*) – defaults to `None`

    Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.

- **similarity** – defaults to `<function dot at 0x7fed30c67ca0>`

    Similarity measure to compare documents embeddings and query embedding (similarity.cosine or similarity.dot).



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
...    on = ["title", "article"],
...    k = 2,
...    path = "test_dpr.pkl"
... )

>>> ranker.add(documents=documents)
DPR ranker
     on: title, article
     k: 2
     similarity: dot
     embeddings stored at: test_dpr.pkl

>>> print(ranker(q="Paris", documents=documents, k=2))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 74.02353,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 68.80651,
  'title': 'Eiffel tower'}]
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

    - **documents**     (*list*)    
    
???- note "dump_embeddings"

    Dump embeddings to the selected directory.

    **Parameters**

    - **embeddings**    
    - **path**     (*str*)    
        Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.
    
???- note "embs"

    Computes and returns embeddings of input documents.

    **Parameters**

    - **documents**     (*list*)    
    
???- note "load_embeddings"

    Load embeddings from an existing directory.

    - **path**     (*str*)    
        Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.
    
