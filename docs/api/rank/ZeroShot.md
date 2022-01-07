# ZeroShot

ZeroShot classifier for ranking. Zero shot does not pre-compute embeddings, it needs the fields to rank the input documents.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **encoder**

    Zero shot classifier to use for ranking

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **multi_class** (*bool*) – defaults to `True`

    If more than one candidate label can be correct, pass multi_class=True to calculate each class independently.



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
...     on = ["title", "article"],
...     k = 2,
... )

>>> ranker
Zero Shot Classifier
     model: typeform/distilbert-base-uncased-mnli
     on: title, article
     k: 2
     multi class: True

>>> print(ranker(q="Paris", documents=documents))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'similarity': 0.44725707173347473,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'similarity': 0.31512799859046936,
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

