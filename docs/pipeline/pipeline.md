# Pipeline

Cherche replaces the operators `+` (pipeline), `|` (union), and `|` (intersection) to build pipelines efficiently.

## Pipeline `+`

`+` is the operator dedicated to pipelines. The first model in a pipeline should always be a retriever.

Here is a pipeline made of a retriever and a ranker:

```python
>>> search = retriever + ranker
>>> search.add(documents)
```

The pipeline allows to map document indexes to their content (not needed with Elasticsearch):

```python
>>> search = retriever + ranker + documents
>>> search.add(documents)
```

Pipeline for question answering (mapping ids to documents is mandatory for question answering unless using Elasticsearch):

```python
>>> search_qa = retriever + ranker + documents + question_answering
>>> search.add(documents)
```

Pipeline for summarization (mapping ids to documents is mandatory for summarization unless using Elasticsearch):

```python
>>> search_summarize = retriever + ranker + documents + summarize
>>> search.add(documents)
```

Under the hood the `+` operator calls `compose.Pipeline`.

```python
>>> from cherche import compose

>>> search = compose.Pipeline([retriever, ranker])
>>> search.add(documents)
# is the same as
>>> search = retriever + ranker
>>> search.add(documents)
```

## Union `|`

The union operator `|` improves neural search recall by gathering documents retrieved by multiple models. The union will avoid duplicate documents and keep the first one. The first documents out of the union will be from the first model; the next ones will be from the second model. This strategy allows prioritizing one model or pipeline over another. It may make sense to create a union between two separate pipelines, with the first one having the highest precision and the second one having better recall, like a spare tire.

Union of two retrievers

```python
>>> search = retriever_a | retriver_b
>>> search.add(documents)
```

Union of two retrievers folowed by a ranker

```python
>>> search = (retriever_a | retriver_b) + ranker
>>> search.add(documents)
```

Union of two rankers

```python
>>> search = retriever + (ranker_a | ranker_b)
>>> search.add(documents)
```

Union of two pipelines

```python
>>> search = (retriever_a + ranker_a) | (retriever_b + ranker_b)
>>> search.add(documents)
```

Union of three pipelines

```python
>>> search = (retriever_a + ranker_a) | (retriever_b + ranker_b) | retriever_c
>>> search.add(documents)
```

## Intersection `&`

The intersection operator improves the precision of the model by filtering documents on the intersection of proposed candidates of retrievers and rankers.

Intersection of two retrievers

```python
>>> search = retriever_a & retriver_b
>>> search.add(documents)
```

The intersection of two retrievers followed by a ranker:

```python
>>> search = (retriever_a & retriver_b) + ranker
>>> search.add(documents)
```

Intersection of two rankers:

```python
>>> search = retriever + (ranker_a & ranker_b)
>>> search.add(documents)
```

Intersection of two pipelines:

```python
>>> search = (retriever_a + ranker_a) & (retriever_b + ranker_b)
>>> search.add(documents)
```

Intersection of three pipelines:

```python
>>> search = (retriever_a + ranker_a) & (retriever_b + ranker_b) & retriever_c
>>> search.add(documents)
```

## Let's create a fancy pipeline

Here we create a pipeline from the union of two distinct pipelines. The first part of the union improves precision, and the second improves recall. We can use the Semanlink dataset to feed our neural search pipeline.

```python
>>> search = (retriever_a + ranker_a) | (retriever_b + ranker_b)
>>> search.add(documents)
```

And here is the code:

```python
>>> from cherche import retrieve, rank, data
>>> from sentence_transformers import SentenceTransformer
>>> from sklearn.feature_extraction.text import TfidfVectorizer

>>> documents, _ = data.arxiv_tags(arxiv_title=True, arxiv_summary=False, comment=False)

>>> ranker = rank.Encoder(
...    key = "uri",
...    on = ["prefLabel_text", "altLabel_text"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 10,
...    path = "semanlink.pkl"
... )

>>> precision = retrieve.Flash(
...    key = "uri",
...    on = ["prefLabel", "altLabel"], 
...    k = 30,
... ) + ranker

>>> recall = retrieve.TfIdf(
...    key = "uri",
...    on = ["prefLabel_text", "altLabel_text"], 
...    documents = documents,
...    tfidf = TfidfVectorizer(lowercase=True, min_df=1, max_df=0.9, ngram_range=(3, 7), analyzer="char"), 
...    k = 10,
... ) + ranker

>>> search = precision | recall
>>> search.add(documents)

>>> search("Knowledge Base Embedding By Cooperative Knowledge Distillation")
[{'uri': 'http://www.semanlink.net/tag/knowledge_graph_embeddings'},
 {'uri': 'http://www.semanlink.net/tag/text_kg_and_embeddings'},
 {'uri': 'http://www.semanlink.net/tag/text_aware_kg_embedding'},
 {'uri': 'http://www.semanlink.net/tag/bert_kb'},
 {'uri': 'http://www.semanlink.net/tag/knowledge_graph_augmented_language_models'},
 {'uri': 'http://www.semanlink.net/tag/knowledge_graph'},
 {'uri': 'http://www.semanlink.net/tag/kg_and_nlp'},
 {'uri': 'http://www.semanlink.net/tag/knowledge_augmented_language_models'},
 {'uri': 'http://www.semanlink.net/tag/word_embedding'},
 {'uri': 'http://www.semanlink.net/tag/generative_adversarial_network'}]
```

### Fancy pipeline with a question answering model

```python
>>> from cherche import qa
>>> from transformers import pipeline

>>> question_answering = qa.QA(
...    model = pipeline("question-answering", 
...         model = "deepset/roberta-base-squad2", 
...         tokenizer = "deepset/roberta-base-squad2"
...    ),
...    k = 2,
...    on = ["title", "comment"],
... )

>>> search_qa = search + documents + question_answering

>>> search_qa("What are CNN ?")
[{'start': 155,
  'end': 218,
  'answer': 'CNN use convolutions over the input layer to compute the output',
  'qa_score': 0.14450952410697937,
  'altLabel': ['CNN', 'Convnet', 'Convolutional neural networks', 'Convnets'],
  'comment': 'Feed-forward artificial neural network where the individual neurons are tiled in such a way that they respond to overlapping regions in the visual field. CNN use convolutions over the input layer to compute the output. Widely used models for image and video recognition.\r\n\r\nMain assumption: Data are compositional, they are formed of patterns that are:\r\n\r\n- Local\r\n- Stationary\r\n- Multi-scale (hierarchical)\r\n\r\nConvNets leverage the compositionality structure: They extract compositional features and feed them to classifier, recommender, etc (end-to-end).',
  'uri': 'http://www.semanlink.net/tag/convolutional_neural_network',
  'broader_altLabel': ['Artificial neural network', 'ANN', 'NN'],
  'broader_altLabel_text': 'Artificial neural network ANN NN',
  'altLabel_text': 'CNN Convnet Convolutional neural networks Convnets'},
 {'start': 1,
  'end': 65,
  'answer': 'fast and space-efficient way of vectorizing categorical features',
  'qa_score': 9.786998271010816e-05,
  'altLabel': ['Hashing trick', 'Feature hashing'],
  'comment': 'fast and space-efficient way of vectorizing categorical features. Applies a hash function to the features to determine their column index\r\n\r\n\r\n\r\n\r\n\r\n',
  'uri': 'http://www.semanlink.net/tag/feature_hashing',
  'altLabel_text': 'Hashing trick Feature hashing'}]
```

### Fancy pipeline with a summarization model

```python
>>> from cherche import summary
>>> from transformers import pipeline

>>> summarizer = summary.Summary(
...    model = pipeline("summarization", 
...         model="sshleifer/distilbart-cnn-12-6", 
...         tokenizer="sshleifer/distilbart-cnn-12-6", 
...         framework="pt"
...    ),
...    on = ["title", "comment"],
... )

>>> search_summarize = search + documents + summarizer

>>> search_summarize("What are CNN ?")
'CNN use convolutions over the input layer to compute the output. Widely used models for image and video recognition. Feed-forward artificial'
```
