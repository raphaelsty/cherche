# eval

Evaluate a pipeline using pairs of query and answers.



## Parameters

- **search**

    Neural search pipeline.

- **query_answers** (*list*)

    Pairs of query and answers.

- **hits_k** (*range*) – defaults to `range(0, 10)`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve, rank, eval
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...     {"label": "Paris is the capital of France .", "tags": "Paris", "uri": "tag:FranceCapital"},
...     {"label": "It is known as the city of lights .", "tags": ["lights", "Paris"], "uri": "tag:ParisLights"},
...     {"label": "The Eiffel Tower can be found in Paris .", "tags": ["Eiffel", "Paris"], "uri": "tag:EiffelTower"},
...     {"label": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"], "uri": "tag:Occitanie"},
...     {"label": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"], "uri": "tag:PinkCity"},
...     {"label": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"], "uri": "tag:ToulouseRugby"},
... ]

>>> query_answers = [
...     ("Paris", [
...          {"label": "Paris is the capital of France .", "tags": "Paris", "uri": "tag:FranceCapital"},
...          {"label": "It is known as the city of lights .", "tags": ["lights", "Paris"], "uri": "tag:ParisLights"},
...          {"label": "The Eiffel Tower can be found in Paris .", "tags": ["Eiffel", "Paris"], "uri": "tag:EiffelTower"},
...      ]),
...     ("Toulouse", [
...          {"label": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"], "uri": "tag:Occitanie"},
...          {"label": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"], "uri": "tag:PinkCity"},
...          {"label": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"], "uri": "tag:ToulouseRugby"},
...      ]),
... ]

>>> retriever = retrieve.Flash(on="tags") | retrieve.TfIdf(on="label")

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "label",
...    k = 4,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker

>>> search = search.add(documents)

>>> print(eval.eval(search=search, query_answers=query_answers, hits_k=range(6)))
{'F1@1': '50.00%',
 'F1@2': '80.00%',
 'F1@3': '100.00%',
 'F1@4': '100.00%',
 'F1@5': '100.00%',
 'Precision': '100.00%',
 'Precision@1': '100.00%',
 'Precision@2': '100.00%',
 'Precision@3': '100.00%',
 'Precision@4': '100.00%',
 'Precision@5': '100.00%',
 'R-Precision': '100.00%',
 'Recall@1': '33.33%',
 'Recall@2': '66.67%',
 'Recall@3': '100.00%',
 'Recall@4': '100.00%',
 'Recall@5': '100.00%'}

>>> print(search("Paris"))
[{'label': 'Paris is the capital of France .',
  'similarity': 0.72453946,
  'tags': 'Paris',
  'uri': 'tag:FranceCapital'},
 {'label': 'The Eiffel Tower can be found in Paris .',
  'similarity': 0.52091306,
  'tags': ['Eiffel', 'Paris'],
  'uri': 'tag:EiffelTower'},
 {'label': 'It is known as the city of lights .',
  'similarity': 0.42550576,
  'tags': ['lights', 'Paris'],
  'uri': 'tag:ParisLights'}]

>>> print(search("Toulouse city"))
[{'label': 'Toulouse has a famous rugby club .',
  'similarity': 0.7541945,
  'tags': ['Toulouse', 'rugby'],
  'uri': 'tag:ToulouseRugby'},
 {'label': 'Toulouse is the capital of Occitanie .',
  'similarity': 0.6744734,
  'tags': ['Toulouse', 'Occitanie'],
  'uri': 'tag:Occitanie'},
 {'label': 'It is known as the pink city .',
  'similarity': 0.42794573,
  'tags': ['Toulouse', 'pink', 'rose'],
  'uri': 'tag:PinkCity'},
 {'label': 'It is known as the city of lights .',
  'similarity': 0.3985046,
  'tags': ['lights', 'Paris'],
  'uri': 'tag:ParisLights'}]
```

## References

1. (Evaluation Metrics For Information Retrieval)[https://amitness.com/2020/08/information-retrieval-evaluation/#1-precisionk]
