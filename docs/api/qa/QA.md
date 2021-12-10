# QA

Question Answering model.



## Parameters

- **model**

- **on** (*str*)

- **k** (*int*) – defaults to `None`



## Examples

```python
>>> from pprint import pprint as print
>>> from transformers import pipeline
>>> from cherche import qa

>>> model = qa.QA(
...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
...     on = "title",
...  )

>>> model
Question Answering
     model: deepset/roberta-base-squad2
     on: title

>>> documents = [
...     {"url": "ckb/github.com", "title": "Github library with PyTorch and Transformers .", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "Github Library with PyTorch .", "date": "22-11-2021"},
...     {"url": "blp/github.com", "title": "Github Library with Pytorch and Transformers .", "date": "22-11-2020"},
... ]

>>> print(model(q="What is used with Transformers?", documents=documents, k=2))
[{'answer': 'Github Library',
  'date': '22-11-2020',
  'end': 14,
  'qa_score': 0.2863011956214905,
  'start': 0,
  'title': 'Github Library with Pytorch and Transformers .',
  'url': 'blp/github.com'},
 {'answer': 'Github library',
  'date': '10-11-2021',
  'end': 14,
  'qa_score': 0.2725629210472107,
  'start': 0,
  'title': 'Github library with PyTorch and Transformers .',
  'url': 'ckb/github.com'}]
```

## Methods

???- note "__call__"

    Question answering main method.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
