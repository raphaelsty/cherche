# eval

Evaluate a pipeline using query and answers.



## Parameters

- **search**

- **query_answers** (*list*)

- **on** (*str*)



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

>>> retriever = retrieve.Flash(on="tags")

>>> ranker = rank.Encoder(
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    on = "label",
...    k = 3,
...    path = "encoder.pkl"
... )

>>> search = retriever + ranker

>>> search = search.add(documents)

>>> eval.eval(search=search, query_answers=query_answers, on="label")
{'Precision@1': '100.00%', 'Precision@2': '100.00%', 'Precision@3': '100.00%', 'R-Precision': '100.00%', 'Precision': '100.00%'}

>>> print(search("Paris"))
[{'cosine_distance': 0.2754606008529663,
  'label': 'Paris is the capital of France .',
  'tags': 'Paris',
  'uri': 'tag:FranceCapital'},
 {'cosine_distance': 0.4790869355201721,
  'label': 'The Eiffel Tower can be found in Paris .',
  'tags': ['Eiffel', 'Paris'],
  'uri': 'tag:EiffelTower'},
 {'cosine_distance': 0.5744942128658295,
  'label': 'It is known as the city of lights .',
  'tags': ['lights', 'Paris'],
  'uri': 'tag:ParisLights'}]

>>> print(search("Toulouse"))
[{'cosine_distance': 0.2715187072753906,
  'label': 'Toulouse has a famous rugby club .',
  'tags': ['Toulouse', 'rugby'],
  'uri': 'tag:ToulouseRugby'},
 {'cosine_distance': 0.3184468746185303,
  'label': 'Toulouse is the capital of Occitanie .',
  'tags': ['Toulouse', 'Occitanie'],
  'uri': 'tag:Occitanie'},
 {'cosine_distance': 0.7183035612106323,
  'label': 'It is known as the pink city .',
  'tags': ['Toulouse', 'pink', 'rose'],
  'uri': 'tag:PinkCity'}]
```

