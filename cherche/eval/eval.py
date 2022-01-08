__all__ = ["eval"]

import collections

from river import stats


def eval(search, query_answers: list, hits_k: range = range(10)) -> dict:
    """Evaluate a pipeline using pairs of query and answers.

    Parameters
    ----------
    search
        Neural search pipeline.
    query_answers
        Pairs of query and answers.

    Examples
    --------

    >>> from pprint import pprint as print
    >>> from cherche import retrieve, rank, eval
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...     {"label": "Paris is the capital of France .", "tags": "Paris", "uri": "tag:FranceCapital"},
    ...     {"label": "It is known as the city of lights .", "tags": ["lights", "Paris"], "uri": "tag:ParisLights"},
    ...     {"label": "The Eiffel Tower can be found in Paris .", "tags": ["Eiffel", "Paris"], "uri": "tag:EiffelTower"},
    ...     {"label": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"], "uri": "tag:Occitanie"},
    ...     {"label": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"], "uri": "tag:PinkCity"},
    ...     {"label": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"], "uri": "tag:ToulouseRugby"},
    ... ]

    >>> query_answers = [
    ...     ("Paris", [
    ...          {"uri": "tag:FranceCapital"},
    ...          {"uri": "tag:ParisLights"},
    ...          {"uri": "tag:EiffelTower"},
    ...     ]),
    ...     ("Toulouse", [
    ...          {"uri": "tag:Occitanie"},
    ...          {"uri": "tag:PinkCity"},
    ...          {"uri": "tag:ToulouseRugby"},
    ...     ]),
    ... ]

    >>> retriever = (
    ...     retrieve.Flash(key="uri", on="tags") |
    ...     retrieve.TfIdf(key="uri", on="label", documents=documents)
    ... )

    >>> ranker = rank.Encoder(
    ...    key="uri",
    ...    on = "label",
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    k = 4,
    ...    path = "encoder.pkl"
    ... )

    >>> search = retriever + ranker

    >>> search.add(documents)
    Union
    -----
    Flash retriever
         key: uri
         on: tags
         documents: 8
    TfIdf retriever
         key: uri
         on: label
         documents: 6
    -----
    Encoder ranker
         key: uri
         on: label
         k: 4
         similarity: cosine
         embeddings stored at: encoder.pkl

    >>> print(eval.eval(search=search, query_answers=query_answers, hits_k=range(6)))
    {'F1@1': '50.00%',
     'F1@2': '80.00%',
     'F1@3': '100.00%',
     'F1@4': '100.00%',
     'F1@5': '100.00%',
     'Precision': '100.00%',
     'Precision@1': '100.00%',
     'Precision@2': '100.00%',
     'Precision@3': '100.00%',
     'Precision@4': '100.00%',
     'Precision@5': '100.00%',
     'R-Precision': '100.00%',
     'Recall@1': '33.33%',
     'Recall@2': '66.67%',
     'Recall@3': '100.00%',
     'Recall@4': '100.00%',
     'Recall@5': '100.00%'}

    >>> search += documents

    >>> print(search("Paris"))
    [{'label': 'Paris is the capital of France .',
      'similarity': 0.72453946,
      'tags': 'Paris',
      'uri': 'tag:FranceCapital'},
     {'label': 'The Eiffel Tower can be found in Paris .',
      'similarity': 0.52091306,
      'tags': ['Eiffel', 'Paris'],
      'uri': 'tag:EiffelTower'},
     {'label': 'It is known as the city of lights .',
      'similarity': 0.42550576,
      'tags': ['lights', 'Paris'],
      'uri': 'tag:ParisLights'}]

    >>> print(search("Toulouse city"))
    [{'label': 'Toulouse has a famous rugby club .',
      'similarity': 0.7541945,
      'tags': ['Toulouse', 'rugby'],
      'uri': 'tag:ToulouseRugby'},
     {'label': 'Toulouse is the capital of Occitanie .',
      'similarity': 0.6744734,
      'tags': ['Toulouse', 'Occitanie'],
      'uri': 'tag:Occitanie'},
     {'label': 'It is known as the pink city .',
      'similarity': 0.42794573,
      'tags': ['Toulouse', 'pink', 'rose'],
      'uri': 'tag:PinkCity'},
     {'label': 'It is known as the city of lights .',
      'similarity': 0.3985046,
      'tags': ['lights', 'Paris'],
      'uri': 'tag:ParisLights'}]

    References
    ----------
    1. [Evaluation Metrics For Information Retrieval](https://amitness.com/2020/08/information-retrieval-evaluation/#1-precisionk)

    """
    precision = collections.defaultdict(lambda: stats.Mean())
    recall = collections.defaultdict(lambda: stats.Mean())
    f1 = collections.defaultdict(lambda: stats.Mean())
    global_precision = stats.Mean()
    r_precision = stats.Mean()

    for q, golds in query_answers:
        candidates = search(q=q)

        for candidate in candidates:
            if "similarity" in candidate:
                candidate.pop("similarity")

        # Precision @ k
        for k in hits_k:
            for candidate in candidates[:k]:
                precision[k].update(1) if candidate in golds else precision[k].update(0)

        # Recall @ k
        for k in hits_k:
            if k == 0:
                continue
            positives = 0
            for candidate in candidates[:k]:
                if candidate in golds:
                    positives += 1
            recall[k].update(positives / len(golds)) if positives > 0 else recall[k].update(0)

        for k, candidate in enumerate(candidates):
            global_precision.update(1) if candidate in golds else global_precision.update(0)

        # R-Precision
        relevant = 0
        for candidate in candidates[: len(golds)]:
            if candidate in golds:
                relevant += 1
        r_precision.update(relevant / len(golds) if relevant > 0 else 0)

    # F1 @ k
    for k in hits_k:
        if k == 0:
            continue
        f1[k] = (
            (2 * precision[k].get() * recall[k].get()) / (precision[k].get() + recall[k].get())
            if (precision[k].get() + recall[k].get()) > 0
            else 0
        )

    metrics = {f"Precision@{k}": f"{metric.get():.2%}" for k, metric in precision.items()}
    metrics.update({f"Recall@{k}": f"{metric.get():.2%}" for k, metric in recall.items()})
    metrics.update({f"F1@{k}": f"{metric:.2%}" for k, metric in f1.items()})
    metrics.update({"R-Precision": f"{r_precision.get():.2%}"})
    metrics.update({"Precision": f"{global_precision.get():.2%}"})
    return metrics
