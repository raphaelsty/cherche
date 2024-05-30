# evaluation

Evaluation function



## Parameters

- **search**

    Search function.

- **query_answers** (*List[Tuple[str, List[Dict[str, str]]]]*)

    List of tuples (query, answers).

- **hits_k** (*range*) – defaults to `range(1, 6)`

    List of k to compute precision, recall and F1.

- **batch_size** (*Optional[int]*) – defaults to `None`

    Batch size.

- **k** (*Optional[int]*) – defaults to `None`

    Number of documents to retrieve.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import data, evaluate, retrieve
>>> from lenlp import sparse

>>> documents, query_answers = data.arxiv_tags(
...    arxiv_title=True, arxiv_summary=False, comment=False
... )

>>> search = retrieve.TfIdf(
...     key="uri",
...     on=["prefLabel_text", "altLabel_text"],
...     documents=documents,
...     tfidf=sparse.TfidfVectorizer(normalize=True, ngram_range=(3, 7), analyzer="char"),
... ) + documents

>>> scores = evaluate.evaluation(search=search, query_answers=query_answers, k=10)

>>> print(scores)
{'F1@1': '26.52%',
 'F1@2': '29.41%',
 'F1@3': '28.65%',
 'F1@4': '26.85%',
 'F1@5': '25.19%',
 'Precision@1': '63.06%',
 'Precision@2': '43.47%',
 'Precision@3': '33.12%',
 'Precision@4': '26.67%',
 'Precision@5': '22.55%',
 'R-Precision': '26.95%',
 'Recall@1': '16.79%',
 'Recall@2': '22.22%',
 'Recall@3': '25.25%',
 'Recall@4': '27.03%',
 'Recall@5': '28.54%'}
```

