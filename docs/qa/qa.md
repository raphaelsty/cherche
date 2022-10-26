# Question Answering

The `qa.QA` module integrates an extractive question answering model to the neural search pipeline. The `qa.Qa` module is compatible with [Hugging Face](https://huggingface.co/models?pipeline_tag=question-answering). The `qa.QA` model extracts the most likely spans to answer the user's question from a list of documents. The neural search pipeline filters the whole corpus to reduce the search of spans to a small number of documents and significantly accelerates the search for top answers. However, even when filtering the corpus, Question answering models are relatively slow using CPU and require a GPU to get decent response times.

## On, k

The `on` parameter allows selecting the field(s) on which the question-answering model will extract the answer. When selecting multiple fields via the `on` parameter, we will concatenate them.

The parameter `k` allows retrieving top `k` answers.

## qa_score, answer

`qa.QA` model returns the candidate's documents, the `qa_score` and the `answer` fields. The `answer` field contains the span that is likely to answer the question. The `qa_score` (higher is better) is associated
with the span. The question-answering model orders answer by score.

## Documents

The pipeline must provide the documents and not only the identifiers to the question answering model such as (except for Elasticsearch, which retrieve documents by default):

```python
search = pipeline + documents + question_answering
```

## Quick Start

```python
>>> from cherche import data, rank, retrieve, qa
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k = 30)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = "article",
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 3,
... )

>>> question_answering = qa.QA(
...    model = pipeline("question-answering",
...         model = "deepset/roberta-base-squad2",
...         tokenizer = "deepset/roberta-base-squad2"
...    ),
...    on = "article",
...    k = 2,
... )

>>> search = retriever + ranker + documents + question_answering
>>> search.add(documents)
# Paris Saint-Germain is the answer.
>>> search("What is the name of the football club of Paris?")
[{'start': 18,
  'end': 37,
  'answer': 'Paris Saint-Germain',
  'qa_score': 0.9848363399505615,
  'id': 20,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'The football club Paris Saint-Germain and the rugby union club Stade Français are based in Paris.',
  'similarity': 0.7104821},
 {'start': 15,
  'end': 17,
  'answer': '12',
  'qa_score': 0.015906214714050293,
  'id': 16,
  'title': 'Paris',
  'url': 'https://en.wikipedia.org/wiki/Paris',
  'article': 'Paris received 12.',
  'similarity': 0.46774143},
]
```
