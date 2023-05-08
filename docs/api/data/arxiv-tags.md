# arxiv_tags

Semanlink tags arXiv documents. The objective of this dataset is to evaluate a neural search pipeline for automatic tagging of arXiv documents. This function returns the set of tags and the pairs arXiv documents and tags.



## Parameters

- **arxiv_title** (*bool*) – defaults to `True`

    Include title of the arxiv paper inside the query.

- **arxiv_summary** (*bool*) – defaults to `True`

    Include summary of the arxiv paper inside the query.

- **comment** (*bool*) – defaults to `False`

    Include comment of the arxiv paper inside the query.

- **broader_prefLabel_text** (*bool*) – defaults to `True`

    Include broader_prefLabel as a text field.

- **broader_altLabel_text** (*bool*) – defaults to `True`

    Include broader_altLabel_text as a text field.

- **prefLabel_text** (*bool*) – defaults to `True`

    Include prefLabel_text as a text field.

- **altLabel_text** (*bool*) – defaults to `True`

    Include altLabel_text as a text field.



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import data

>>> documents, query_answers = data.arxiv_tags()

>>> print(list(documents[0].keys()))
['prefLabel',
 'type',
 'broader',
 'creationTime',
 'creationDate',
 'comment',
 'uri',
 'broader_prefLabel',
 'broader_related',
 'broader_prefLabel_text',
 'prefLabel_text']
```

