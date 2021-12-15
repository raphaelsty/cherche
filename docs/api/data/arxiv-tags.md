# arxiv_tags

Semanlink tags arxiv dataset. The objective of this dataset is to evaluate a neural search pipeline for automatic tagging of arxiv documents. This function returns the set of tags and the pairs arxiv documents and tags.

Parameters --------- arxiv_title     Include title of the arxiv paper inside the query. arxiv_summary     Include summary of the arxiv paper inside the query. comment     Include comment of the arxiv paper inside the query. broader_prefLabel_text     Include broader_prefLabel as a text field. broader_altLabel_text     Include broader_altLabel_text as a text field. prefLabel_text     Include prefLabel_text as a text field. altLabel_text     Include altLabel_text as a text field.

## Parameters

- **arxiv_title** (*bool*) – defaults to `True`

- **arxiv_summary** (*bool*) – defaults to `True`

- **comment** (*bool*) – defaults to `False`

- **broader_prefLabel_text** (*bool*) – defaults to `True`

- **broader_altLabel_text** (*bool*) – defaults to `True`

- **prefLabel_text** (*bool*) – defaults to `True`

- **altLabel_text** (*bool*) – defaults to `True`



## Examples

```python
>>> from pprint import pprint as print
>>> from cherche import data

>>> documents, query_answers = data.arxiv_tags()

>>> print(list(documents[0].keys()))
['prefLabel',
 'describedBy',
 'type',
 'broader',
 'creationTime',
 'altLabel',
 'creationDate',
 'comment',
 'related',
 'sameAs',
 'homepage',
 'weblog',
 'linkToMusicBrainz',
 'publish',
 'subject',
 'seeAlso',
 'wikipage-en',
 'uri',
 'broader_prefLabel',
 'broader_altLabel',
 'broader_related',
 'broader_prefLabel_text',
 'broader_altLabel_text',
 'prefLabel_text',
 'altLabel_text']
```

