# QA

Question Answering model.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to answer to the question.

- **model**

    Hugging Face question answering model available [here](https://huggingface.co/models?pipeline_tag=question-answering).

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from transformers import pipeline
>>> from cherche import qa
>>> from pprint import pprint as print

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> model = qa.QA(
...     model = pipeline("question-answering", model = "deepset/roberta-base-squad2", tokenizer = "deepset/roberta-base-squad2"),
...     on = ["title", "article"],
...     k = 2,
...  )

>>> model
Question Answering
    on: title, article

>>> print(model(q="Where is the Eiffel tower?", documents=documents))
[{'answer': 'Paris',
  'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'end': 43,
  'qa_score': 0.9743021130561829,
  'start': 38,
  'title': 'Eiffel tower'},
 {'answer': 'Paris',
  'article': 'This town is the capital of France',
  'author': 'Wiki',
  'end': 5,
  'qa_score': 0.0003580129996407777,
  'start': 0,
  'title': 'Paris'}]
```

## Methods

???- note "__call__"

    Question answering main method.

    **Parameters**

    - **q**     (*str*)    
    - **documents**     (*list*)    
    - **kwargs**    
    
