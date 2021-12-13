# ZeroShot

ZeroShot classifier for ranking.



## Parameters

- **encoder**

- **on** (*str*)

- **k** (*int*) – defaults to `None`

- **multi_class** (*bool*) – defaults to `True`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import rank
>>> from transformers import pipeline
>>> from sentence_transformers import SentenceTransformer

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> ranker = rank.ZeroShot(
...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
...     on = "article",
...     k = 2,
... )

>>> ranker
Zero Shot Classifier
     model: typeform/distilbert-base-uncased-mnli
     on: article
     k: 2
     multi class: True

>>> print(ranker(q="Paris", documents=documents))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 0.4519128203392029,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 0.33974456787109375,
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Rank inputs documents based on query.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
???- note "add"

    Zero shot do not pre-compute embeddings.

    **Parameters**

    - **documents**    
    
## References

1. [New pipeline for zero-shot text classification](https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681)
2. [NLI models](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads)

