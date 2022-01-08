# Encoder

SentenceBert Ranker.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **key** (*str*)

    Field identifier of each document.

- **encoder**

    Encoding function dedicated to documents and query.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **path** (*str*) – defaults to `None`

    Path to the file dedicated to storing the embeddings. The ranker will read this file if it already exists to load the embeddings and will update it when documents are added.

- **similarity** (*<module 'cherche.similarity' from '/Users/raphaelsourty/opt/miniconda3/envs/cherche/lib/python3.8/site-packages/cherche/similarity/__init__.py'>*) – defaults to `<function cosine at 0x7f9c81ed63a0>`

    Similarity measure to compare documents embeddings and query embedding (similarity.cosine or similarity.dot).



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
...    path = "encoder.pkl"
... )

>>> ranker.add(documents=documents)
Encoder ranker
     key: id
     on: title, article
     k: 2
     similarity: cosine
     embeddings stored at: encoder.pkl

>>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
[{'id': 0, 'similarity': 0.66051394}, {'id': 1, 'similarity': 0.5142564}]

>>> print(ranker(q="Paris", documents=documents))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.66051394,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.5142564,
  'title': 'Eiffel tower'}]

>>> ranker += documents

>>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.66051394,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.5142564,
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
    
