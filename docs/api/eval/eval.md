# eval

Evaluate a pipeline using query and answers.



## Parameters

- **search**

- **query_answers** (*list*)

- **on** (*str*)



## Examples

```python
>>> from cherche import retrieve, rank, eval
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...     {"document": "Paris is the capital of France .", "tags": ["Paris", "paris"]},
...     {"document": "It is known as the city of lights .", "tags": ["lights", "paris"]},
...     {"document": "It is known as the city of lights .", "tags": ["lights", "paris"]},
...     {"document": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"]},
...     {"document": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"]},
...     {"document": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"]},
... ]

>>> query_answers = [
...     ("Paris", [
...          {"document": "Paris is the capital of France .", "tags": ["Paris", "paris"]},
...          {"document": "It is known as the city of lights .", "tags": ["lights", "paris"]},
...          {"document": "The Eiffel Tower can be found in Paris .", "tags": ["Eiffel", "paris"]},
...      ]),
...     ("Toulouse", [
...          {"document": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"]},
...          {"document": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"]},
...          {"document": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"]},
...      ]),
... ]

>>> retriever = retrieve.Flash(on="tags")

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "document",
...    k = 3,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker

>>> search = search.add(documents)

>>> eval.eval(search=search, query_answers=query_answers, on="document")
{'Precision@1': '100.00%', 'Precision@2': '100.00%', 'Precision@3': '100.00%', 'R-Precision': '83.33%', 'Precision': '100.00%'}
```

