# PRF

Pseudo (or blind) Relevance-Feedback module. The Query-Augmentation method applies a fast document retrieving method and then extracts keywords from relevant documents. Thus, we have to retrieve top words from relevant documents to give a proper augmentation of a given query.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use for fitting the spelling corrector on.

- **documents** (*list*)

- **tf** (*sklearn.feature_extraction.text.CountVectorizer*) – defaults to `sparse.TfidfVectorizer()`

    defaults to sklearn.feature_extraction.text.sparse.TfidfVectorizer. If you want to implement your own tf, it needs to follow the sklearn base API and provides the `transform` `fit_transform` and `get_feature_names_out` methods. See sklearn documentation for more information.

- **nb_docs** (*int*) – defaults to `5`

    Number of documents from which to retrieve top-terms.

- **nb_terms_per_doc** (*int*) – defaults to `3`

    Number of terms to extract from each top documents retrieved.


## Attributes

- **type**


## Examples

```python
>>> from cherche import query, data

>>> documents = data.load_towns()

>>> prf = query.PRF(
...     on=["title", "article"],
...     nb_docs=8, nb_terms_per_doc=1,
...     documents=documents
... )

>>> prf
Query PRF
    on       : title, article
    documents: 8
    terms    : 1

>>> prf(q="Europe")
'Europe art metro space science bordeaux paris university significance'

>>> prf(q=["Europe", "Paris"])
['Europe art metro space science bordeaux paris university significance', 'Paris received paris club subway billion source tour tournament']
```

## Methods

???- note "__call__"

    Augment a given query with new terms.

    **Parameters**

    - **q**     (*Union[List[str], str]*)    
    - **kwargs**    
    
## References

1. [Relevance feedback and pseudo relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-pseudo-relevance-feedback-1.html)
2. [Blind Feedback](https://en.wikipedia.org/wiki/Relevance_feedback#Blind_feedback)

