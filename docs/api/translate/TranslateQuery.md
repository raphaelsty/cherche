# TranslateQuery

Translation module using Hugging Face pre-trained models.



## Parameters

- **translation**

    Hugging Face translation pipeline.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import translate, retrieve
>>> from transformers import pipeline

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> query_translation = translate.TranslateQuery(
...     translation = pipeline("translation_fr_to_en", model = "Helsinki-NLP/opus-mt-fr-en"),
... )

>>> search = query_translation + retrieve.TfIdf(key = "id", on = "article", k=1, documents=documents)

>>> query_translation("tour eiffel")
'eiffel tower'

>>> print(search("eiffel tower"))
[{'id': 1}]
```

## Methods

???- note "__call__"

    Call self as a function.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
## References

1. [Hugging Face](https://huggingface.co/models?pipeline_tag=translation)

