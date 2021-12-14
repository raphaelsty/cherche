# TfIdf

TfIdf retriever based on cosine similarities.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use to match the query to the documents.

- **k** (*int*) – defaults to `None`

    Number of documents to retrieve. Default is None, i.e all documents that match the query will be retrieved.

- **tfidf** (*sklearn.feature_extraction.text.TfidfVectorizer*) – defaults to `None`

    TfidfVectorizer class of Sklearn to create a custom TfIdf retriever.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import retrieve

>>> retriever = retrieve.TfIdf(on=["title", "article"], k=3)

>>> documents = [
...    {"title": "Paris", "article": "This town is the capital of France", "author": "Wiki"},
...    {"title": "Eiffel tower", "article": "Eiffel tower is based in Paris", "author": "Wiki"},
...    {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wiki"},
... ]

>>> retriever = retriever.add(documents=documents)

>>> retriever
TfIdf retriever
     on: title, article
     documents: 3

>>> print(retriever(q="paris"))
[{'article': 'This town is the capital of France',
  'author': 'Wiki',
  'title': 'Paris'},
 {'article': 'Eiffel tower is based in Paris',
  'author': 'Wiki',
  'title': 'Eiffel tower'}]
```

## Methods

???- note "__call__"

    Retrieve the right document.

    **Parameters**

    - **q**     (*str*)    
    
???- note "add"

    Add documents to the retriever.

    **Parameters**

    - **documents**     (*list*)    
    
## References

1. [sklearn.feature_extraction.text.TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
2. [Python: tf-idf-cosine: to find document similarity](https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity)

