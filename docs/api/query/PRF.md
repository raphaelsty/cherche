# PRF

Pseudo (or blind) Relevance-Feedback module. Query-Augmentation method consisting of applying a fast document retrieving method, then extracting keywords from top documents. The main principle of this method is that the top documents from any working ranking method should give at least great results (ie: the user almost always considers the first documents as relevant). Thus, we juste have to retrieve top-words from relevant documents to give a proper augmentation of a given query.



## Parameters

- **on** (*Union[str, list]*)

    Fields to use for fitting the spelling corrector on.

- **documents** (*list*)

- **tf** (*sklearn.feature_extraction.text.CountVectorizer*) – defaults to `TfidfVectorizer()`

    defaults to sklearn.feature_extraction.text.TfidfVectorizer. If you want to implement your own tf, it needs to follow the sklearn base API and provides the `transform` `fit_transform` and `get_feature_names_out` methods. See sklearn documentation for more information.

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

>>> prf = query.PRF(on=["title", "article"], nb_docs=8, nb_terms_per_doc=1, documents=documents)

>>> prf
Query PRF
     On: title, article
     Documents: 8
     Terms: 1

>>> prf(q="Europe")
'Europe centres metro space art paris bordeaux significance university'
```

## Methods

???- note "__call__"

    Augment a given query with new terms.

    **Parameters**

    - **q**     (*str*)    
    - **kwargs**    
    
## References

1. [Relevance feedback and pseudo relevance feedback](https://nlp.stanford.edu/IR-book/html/htmledition/relevance-feedback-and-pseudo-relevance-feedback-1.html)
2. [Blind Feedback](https://en.wikipedia.org/wiki/Relevance_feedback#Blind_feedback)

