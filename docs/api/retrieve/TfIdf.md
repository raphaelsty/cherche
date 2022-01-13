# TfIdf

TfIdf retriever based on cosine similarities.



## Parameters

- **key** (*str*)

    Field identifier of each document.

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **documents** (*list*)

    Documents in TFIdf retriever are static. The retriever must be reseted to index new documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is `None`, i.e all documents that match the query will be retrieved.

- **tfidf** (*sklearn.feature_extraction.text.TfidfVectorizer*) – defaults to `None`

    TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.


## Attributes

- **type**


## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> documents = [
...    {"id": 0, "title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"id": 1, "title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"id": 2, "title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=3)

>>> retriever
TfIdf retriever
     key: id
     on: title, article
     documents: 3

>>> print(retriever(q="paris"))
[{'id': 0}, {'id': 1}]

>>> retriever += documents

>>> print(retriever(q="paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'id': 0,
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'id': 1,
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Retrieve the right document.

    **Parameters**

    - **q**     (*str*)    
    
## References

1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

