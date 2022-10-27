# DPR

DPR ranks documents using distinct models to encode the query and document contents.



## Parameters

- **key** (*'str'*)

    Field identifier of each document.

- **on** (*'str | list'*)

    Fields to use to match the query to the documents.

- **encoder**

    Encoding function dedicated to documents.

- **query_encoder**

    Encoding function dedicated to the query.

- **k** (*'int | typing.Optionnal'*) – defaults to `None`

    Number of documents to reorder. The default value is None, i.e. all documents will be reordered and returned.

- **similarity** – defaults to `<function dot at 0x16beca040>`

    Similarity measure to compare documents embeddings and query embedding (similarity.cosine or similarity.dot).

- **store** – defaults to `<cherche.rank.base.MemoryStore object at 0x16be3f910>`

- **path** (*'str | typing.Optionnal'*) – defaults to `None`


## Attributes

- **type**


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

>>> ranker = rank.DPR(
...    key = "id",
...    on = ["title", "article"],
...    encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base').encode,
...    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base').encode,
...    k = 2,
... )

>>> ranker.add(documents=documents)
DPR ranker
    key: id
    on: title, article
    k: 2
    similarity: dot
    Embeddings pre-computed: 3

>>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
[{'id': 0, 'similarity': 74.0235366821289},
 {'id': 1, 'similarity': 68.8065185546875}]

>>> print(ranker(q="Paris", documents=documents))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 74.0235366821289,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 68.8065185546875,
  'title': 'Eiffel tower'}]

>>> ranker += documents

>>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 74.0235366821289,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 68.8065185546875,
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Encode inputs query and ranks documents based on the similarity between the query and the selected field of the documents.

    **Parameters**

    - **q**     (*'str'*)    
    - **documents**     (*'list'*)    
    - **kwargs**    
    
???- note "add"

    Pre-compute embeddings and store them at the selected path.

    **Parameters**

    - **documents**     (*'list'*)    
    - **batch_size**     (*'int'*)     – defaults to `64`    
    
???- note "encode"

    Computes documents embeddings.

    **Parameters**

    - **documents**     (*'list'*)    
    
???- note "rank"

    Rank inputs documents ordered by relevance among the top k.

    **Parameters**

    - **query_embedding**     (*'np.ndarray'*)    
    - **documents**     (*'list'*)    
    
