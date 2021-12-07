# Summary

Summarization model. Returns a single summary for inputs documents.



## Parameters

- **model**

- **on** (*str*)

- **min_length** (*int*) – defaults to `5`

- **max_length** (*int*) – defaults to `30`



## Examples

```python
>>> from transformers import pipeline
>>> from cherche import summary

>>> model = summary.Summary(
...    model = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", tokenizer="sshleifer/distilbart-cnn-6-6", framework="pt"),
...    on = "title",
... )

>>> model
Summarization model
     on: title
     min length: 5
     max length: 30

>>> documents = [
...     {"url": "ckb/github.com", "title": "CKB is a Github library with PyTorch and Transformers.", "date": "10-11-2021"},
...     {"url": "mkb/github.com", "title": "MKB Github Library with PyTorch  dedicated to KB.", "date": "22-11-2021"},
...     {"url": "blp/github.com", "title": "BLP is a Github Library with Pytorch and Transformers dedicated to KB.", "date": "22-11-2020"},
... ]

>>> print(model(documents=documents))
CKB is a Github library with Pytorch and Transformers dedicated to KB. MKB Github Library with PyTorch dedicated toKB
```

## Methods

???- note "__call__"

    Summarize input text.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
