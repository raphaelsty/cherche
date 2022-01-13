# Translate

Translation module using Hugging Face pre-trained models.



## Parameters

- **on** (*Union[list, str]*)

    Field to use to retrieve documents.

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

>>> translation = translate.Translate(
...     on = ["title", "article"],
...     translation = pipeline("translation_en_to_fr", model = "t5-small"),
... )

>>> translation
Translation
     on: ['title', 'article']

>>> print(translation(documents=documents))
[{'article': 'Cette ville est la capitale française',
  'author': 'Wiki',
  'id': 0,
  'title': 'Paris'},
 {'article': 'La tour Eiffel est basée à Paris.',
  'author': 'Wiki',
  'id': 1,
  'title': 'Tour Eiffel'},
 {'article': 'Montréal est au Canada.',
  'author': 'Wiki',
  'id': 2,
  'title': 'Montréal'}]

```

 >>> search = (retrieve.TfIdf(key = "id", on = "title", k=1, documents=documents) +
 ...    documents + translation)

 >>> print(search("Paris"))
 [{'article': 'Cette ville est la capitale française',
  'author': 'Wiki',
  'id': 0,
  'title': 'Paris'}]

 >>> print(documents)
 [{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'title': 'Eiffel tower'},
 {'article': 'Montreal is in Canada.',
  'author': 'Wiki',
  'id': 2,
  'title': 'Montreal'}]

## Methods

???- note "__call__"

    Translate documents. Translate all documents per batch and finally update documents.

    **Parameters**

    - **documents**     (*list*)    
    - **kwargs**    
    
## References

1. [Hugging Face](https://huggingface.co/models?pipeline_tag=translation)

