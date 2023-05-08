# Question Answering

The `qa.QA` module is a crucial component of our neural search pipeline, integrating an extractive question answering model that is compatible with [Hugging Face](https://huggingface.co/models?pipeline_tag=question-answering). This model efficiently extracts the most likely answer spans from a list of documents in response to user queries. To further expedite the search process, our neural search pipeline filters the entire corpus and narrows down the search to a few relevant documents, resulting in faster response times for top answers. However, it's worth noting that even with corpus filtering, question answering models can be slow when using a CPU and typically require a GPU to achieve optimal performance.

## Documents

The pipeline must provide the documents and not only the identifiers to the question answering model such as:

```python
search = pipeline + documents + question_answering
```

## Tutorial

```python
>>> from cherche import data, rank, retrieve, qa
>>> from sentence_transformers import SentenceTransformer
>>> from transformers import pipeline

>>> documents = data.load_towns()

>>> retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=100)

>>> ranker = rank.Encoder(
...    key = "id",
...    on = "article",
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 30,
... )

>>> question_answering = qa.QA(
...    model = pipeline("question-answering",
...         model = "deepset/roberta-base-squad2",
...         tokenizer = "deepset/roberta-base-squad2"
...    ),
...    on = "article",
... )

>>> search = retriever + ranker + documents + question_answering
>>> search.add(documents)
>>> answers = search(
...   q=[
...     "What is the name of the football club of Paris?",
...     "What is the speciality of Lyon?"
...   ]
... )

# The answer is Paris Saint-Germain
>>> answers[0][0]
{'id': 20,
 'title': 'Paris',
 'url': 'https://en.wikipedia.org/wiki/Paris',
 'article': 'The football club Paris Saint-Germain and the rugby union club Stade Français are based in Paris.',
 'similarity': 0.6905894,
 'score': 0.9848365783691406,
 'start': 18,
 'end': 37,
 'answer': 'Paris Saint-Germain',
 'question': 'What is the name of the football club of Paris?'}


>>> answers[1][0]
{'id': 52,
'title': 'Lyon',
'url': 'https://en.wikipedia.org/wiki/Lyon',
'article': 'Economically, Lyon is a major centre for banking, as well as for the chemical, pharmaceutical and biotech industries.',
'similarity': 0.64728546,
'score': 0.6952874660491943,
'start': 41,
'end': 48,
'answer': 'banking',
'question': 'What is the speciality of Lyon?'}
```
