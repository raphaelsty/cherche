# Pipeline

Cherche is a tool that provides operators for building pipelines efficiently. The operators it replaces are `+` (pipeline), `|` (union), `&` (intersection), and `*` (voting).

## Pipeline `+`

The `+` operator is used to create pipelines. Here's an example of a pipeline created with a retriever and a ranker:

```python
>>> search = retriever + ranker
>>> search.add(documents)
```

The pipeline allows you to map output indexes to their content, as shown here:

```python
>>> search = retriever + ranker + documents
>>> search.add(documents)
```

When building a pipeline for question answering, mapping ids to documents is mandatory. Here's an example:

```python
>>> search_qa = retriever + ranker + documents + question_answering
>>> search.add(documents)
```

## Union `|`

The `|` operator improves neural search recall by gathering documents retrieved by multiple models. The union operator will avoid duplicate documents and keep the first one. The first documents out of the union will be from the first model, and the subsequent ones will be from the second model. This strategy allows prioritizing one model or pipeline over another. It may make sense to create a union between two separate pipelines, with the first one having the highest precision and the second one having better recall, like a spare tire.

Here are some examples of unions:

Union of two retrievers:

```python
>>> search = retriever_a | retriever_b
>>> search.add(documents)
```

Union of two retrievers followed by a ranker:

```python
>>> search = (retriever_a | retriever_b) + ranker
>>> search.add(documents)
```

Union of two rankers:

```python
>>> search = retriever + (ranker_a | ranker_b)
>>> search.add(documents)
```

Union of two pipelines:

```python
>>> search = (retriever_a + ranker_a) | (retriever_b + ranker_b)
>>> search.add(documents)
```

Union of three pipelines:

```python
>>> search = (retriever_a + ranker_a) | (retriever_b + ranker_b) | retriever_c
>>> search.add(documents)
```

## Intersection `&`

The `&` operator improves the precision of the model by filtering documents on the intersection of proposed candidates of retrievers and rankers.

Here are some examples of intersections:

Intersection of two retrievers:

```python
>>> search = retriever_a & retriever_b
>>> search.add(documents)
```

Intersection of two retrievers followed by a ranker:

```python
>>> search = (retriever_a & retriever_b) + ranker
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

## Voting `*`

The `*` operator improves both the precision and recall of the model by computing the average normalized similarity between the documents.

Here are some examples of voting:

Vote of two retrievers:

```python
>>> search = retriever_a * retriever_b
>>> search.add
```

Vote of two retrievers followed by a ranker:

```python
>>> search = (retriever_a * retriever_b) + ranker
>>> search.add(documents)
```

Vote of two rankers:

```python
>>> search = retriever + (ranker_a * ranker_b)
>>> search.add(documents)
```

Vote of two pipelines:

```python
>>> search = (retriever_a + ranker_a) * (retriever_b + ranker_b)
>>> search.add(documents)
```

Vote of three pipelines:

```python
>>> search = (retriever_a + ranker_a) * (retriever_b + ranker_b) * retriever_c
>>> search.add(documents)
```

## Let's create a pipeline

Here we create a pipeline from the union of two distinct pipelines. The first part of the union improves precision, and the second improves recall. We can use the Semanlink dataset to feed our neural search pipeline.

```python
>>> search = (retriever_a + ranker_a) | (retriever_b + ranker_b)
>>> search.add(documents)
```

And here is the code:

```python
>>> from cherche import retrieve, rank, data
>>> from sentence_transformers import SentenceTransformer
>>> from lenlp import sparse

>>> documents, _ = data.arxiv_tags(arxiv_title=True, arxiv_summary=False, comment=False)

>>> ranker = rank.Encoder(
...    key = "uri",
...    on = ["prefLabel_text", "altLabel_text"],
...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
...    k = 10,
... )

>>> ranker.add(documents)

>>> precision = retrieve.Flash(
...    key = "uri",
...    on = ["prefLabel", "altLabel"],
...    k = 100,
... ).add(documents) + ranker

>>> recall = retrieve.TfIdf(
...    key = "uri",
...    on = ["prefLabel_text", "altLabel_text"],
...    documents = documents,
...    tfidf = sparse.TfidfVectorizer(normalize=True, min_df=1, max_df=0.9, ngram_range=(3, 7), analyzer="char"),
...    k = 100,
... ) + ranker

>>> search = precision | recall

>>> search("Knowledge Base Embedding By Cooperative Knowledge Distillation")
[{'uri': 'http://www.semanlink.net/tag/knowledge_base',
  'similarity': 2.1666666666666665},
 {'uri': 'http://www.semanlink.net/tag/knowledge_distillation',
  'similarity': 0.5},
 {'uri': 'http://www.semanlink.net/tag/embeddings',
  'similarity': 0.3333333333333333},
 {'uri': 'http://www.semanlink.net/tag/knowledge_graph_embeddings',
  'similarity': 0.25},
 {'uri': 'http://www.semanlink.net/tag/knowledge_driven_embeddings',
  'similarity': 0.2},
 {'uri': 'http://www.semanlink.net/tag/hierarchy_aware_knowledge_graph_embeddings',
  'similarity': 0.16666666666666666},
 {'uri': 'http://www.semanlink.net/tag/entity_embeddings',
  'similarity': 0.14285714285714285},
 {'uri': 'http://www.semanlink.net/tag/text_kg_and_embeddings',
  'similarity': 0.125},
 {'uri': 'http://www.semanlink.net/tag/text_aware_kg_embedding',
  'similarity': 0.1111111111111111},
 {'uri': 'http://www.semanlink.net/tag/knowledge_graph_completion',
  'similarity': 0.1},
 {'uri': 'http://www.semanlink.net/tag/knowledge_graph_deep_learning',
  'similarity': 0.09090909090909091},
 {'uri': 'http://www.semanlink.net/tag/combining_knowledge_graphs',
  'similarity': 0.07692307692307693}]
```
