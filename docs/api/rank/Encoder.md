# Encoder

SentenceBert Ranker.



## Parameters

- **on** (*'str | list'*)

    Fields to use to match the query to the documents.

- **key** (*'str'*)

    Field identifier of each document.

- **encoder**

    Encoding function dedicated to documents and query.

- **k** (*'int | typing.Optionnal'*) – defaults to `None`

    Number of documents to reorder. The default value is None, i.e. all documents will be reordered and returned.

- **similarity** – defaults to `<function cosine at 0x16bec7f70>`

    Similarity measure to compare documents embeddings and query embedding (similarity.cosine or similarity.dot).

- **store** – defaults to `<cherche.rank.base.MemoryStore object at 0x16bebabb0>`

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

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["title", "article"],
...    k = 2,
... )

>>> ranker.add(documents=documents)
Encoder ranker
    key: id
    on: title, article
    k: 2
    similarity: cosine
    Embeddings pre-computed: 3

>>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
[{'id': 0, 'similarity': 0.6605141758918762},
 {'id': 1, 'similarity': 0.5142566561698914}]

>>> print(ranker(q="Paris", documents=documents))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.6605141758918762,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.5142566561698914,
  'title': 'Eiffel tower'}]

>>> ranker += documents

>>> print(ranker(q="Paris", documents=[{"id": 0}, {"id": 1}, {"id": 2}]))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'similarity': 0.6605141758918762,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'similarity': 0.5142566561698914,
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Encode input query and ranks documents based on the similarity between the query and the selected field of the documents.

    https://pymilvus.readthedocs.io/en/latest/tutorial.html status, documents = client.get_entity_by_id(collection_name, [id_1, id_2])

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
    
