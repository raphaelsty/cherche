# Pipeline

Neurals search pipeline.



## Parameters

- **models** (*list*)

    List of models of the pipeline.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, rank
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"id": 0, "town": "Paris", "country": "France", "continent": "Europe"},
...    {"id": 1, "town": "Montreal", "country": "Canada", "continent": "North America"},
...    {"id": 2, "town": "Madrid", "country": "Spain", "continent": "Europe"},
... ]

>>> retriever = retrieve.TfIdf(
...     key="id", on=["town", "country", "continent"], documents=documents)

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    key = "id",
...    on = ["town", "country", "continent"],
... )

>>> pipeline = retriever + ranker

>>> pipeline = pipeline.add(documents)

>>> print(pipeline("Paris Europe"))
[{'id': 0, 'similarity': 0.9149576}, {'id': 2, 'similarity': 0.8091332}]

>>> print(pipeline(["Paris", "Europe", "Paris Madrid Europe France Spain"]))
[[{'id': 0, 'similarity': 0.69523287}],
 [{'id': 0, 'similarity': 0.7381397}, {'id': 2, 'similarity': 0.6488539}],
 [{'id': 0, 'similarity': 0.8582063}, {'id': 2, 'similarity': 0.8200009}]]

>>> pipeline = retriever + ranker + documents

>>> print(pipeline("Paris Europe"))
[{'continent': 'Europe',
  'country': 'France',
  'id': 0,
  'similarity': 0.9149576,
  'town': 'Paris'},
 {'continent': 'Europe',
  'country': 'Spain',
  'id': 2,
  'similarity': 0.8091332,
  'town': 'Madrid'}]

>>> print(pipeline(["Paris", "Europe", "Paris Madrid Europe France Spain"]))
[[{'continent': 'Europe',
   'country': 'France',
   'id': 0,
   'similarity': 0.69523287,
   'town': 'Paris'}],
 [{'continent': 'Europe',
   'country': 'France',
   'id': 0,
   'similarity': 0.7381397,
   'town': 'Paris'},
  {'continent': 'Europe',
   'country': 'Spain',
   'id': 2,
   'similarity': 0.6488539,
   'town': 'Madrid'}],
 [{'continent': 'Europe',
   'country': 'France',
   'id': 0,
   'similarity': 0.8582063,
   'town': 'Paris'},
  {'continent': 'Europe',
   'country': 'Spain',
   'id': 2,
   'similarity': 0.8200009,
   'town': 'Madrid'}]]
```

## Methods

???- note "__call__"

    Pipeline main method. It takes a query and returns a list of documents. If the query is a list of queries, it returns a list of list of documents. If the batch_size_ranker, or batch_size_retriever it takes precedence over the batch_size. If the k_ranker, or k_retriever it takes precedence over the k parameter.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **k**     (*Optional[int]*)     – defaults to `None`    
    - **batch_size**     (*Optional[int]*)     – defaults to `None`    
    - **documents**     (*Optional[List[Dict[str, str]]]*)     – defaults to `None`    
    - **kwargs**    
    
???- note "add"

    Add new documents.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
???- note "reset"

