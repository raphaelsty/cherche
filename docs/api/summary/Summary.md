# Summary

Summarization model. Returns a single summary for an input list of documents.



## Parameters

- **model**

    Hugging Face summarization model available [here](https://huggingface.co/models?pipeline_tag=summarization).

- **on** (*Union[str, list]*)

    Fields to summarize.

- **min_length** (*int*) – defaults to `5`

- **max_length** (*int*) – defaults to `30`


## Attributes

- **type**


## Examples

```python
>>> from transformers import pipeline
>>> from cherche import summary

>>> model = summary.Summary(
...    model = pipeline(
...         "summarization",
...         model="sshleifer/distilbart-cnn-6-6",
...         tokenizer="sshleifer/distilbart-cnn-6-6",
...         framework="pt"
...    ),
...    on = ["title", "article"],
... )

>>> model
Summarization model
     on: title, article
     min length: 5
     max length: 30

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> print(model(documents=documents))
Eiffel tower is based in Paris Montreal Montreal Montreal is in Canada. Paris is the capital of the French capital of France Eiff
```

## Methods

???- note "__call__"

    Summarize input text.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
