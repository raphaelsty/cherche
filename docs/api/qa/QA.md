# QA

Question Answering model.



## Parameters

- **model**

    Hugging Face question answering model available [here](https://huggingface.co/models?pipeline_tag=question-answering).

- **on** (*str*)

    Field to use to answer to the question.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.



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
    
