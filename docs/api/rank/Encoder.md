# Encoder

SentenceBert Ranker.



## Parameters

- **encoder**

    Encoding function dedicated to documents and query.

- **on** (*str*)

    Field to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **path** (*str*) – defaults to `None`

    Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.

- **similarity** – defaults to `<function cosine at 0x7fd101bc9b80>`

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
        Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.
    
???- note "load_embeddings"

    Load embeddings from an existing directory.

    - **path**     (*str*)    
        Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.
    
