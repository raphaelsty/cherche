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

>>> ranker = rank.ZeroShot(
...     encoder = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"),
...     on = "title",
...     k = 2,
... )

>>> ranker
Zero Shot Classifier
     model: typeform/distilbert-base-uncased-mnli
     on: title
     k: 2
     multi class: True

>>> documents = [
...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
... ]

>>> print(ranker(q="Transformers", documents=documents))
[{'_zero_shot_score': 0.3513341546058655,
  'date': '22-11-2020',
  'title': 'Github Library with Pytorch and Transformers .',
  'url': 'blp/github.com'},
 {'_zero_shot_score': 0.3513341546058655,
  'date': '10-11-2021',
  'title': 'Github library with PyTorch and Transformers .',
  'url': 'ckb/github.com'}]
```

## Methods

???- note "__call__"

    Rank inputs documents based on query.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
## References

1. [New pipeline for zero-shot text classification](https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681)
2. [NLI models](https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads)

