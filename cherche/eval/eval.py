__all__ = ["eval"]

import collections

from creme import stats


def eval(search, query_answers: list, on: str):
    """Evaluate a pipeline using query and answers.

    Parameters
    ----------

        search: Cherche pipeline.
        query_answers: Pair of query and answers.
        on: Field to use compare retrieved documents and test set.

    Examples
    --------

    >>> from cherche import retrieve, rank, eval
    >>> from sentence_transformers import SentenceTransformer

    >>> documents = [
    ...     {"document": "Paris is the capital of France .", "tags": ["Paris", "paris"]},
    ...     {"document": "It is known as the city of lights .", "tags": ["lights", "paris"]},
    ...     {"document": "It is known as the city of lights .", "tags": ["lights", "paris"]},
    ...     {"document": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"]},
    ...     {"document": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"]},
    ...     {"document": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"]},
    ... ]

    >>> query_answers = [
    ...     ("Paris", [
    ...          {"document": "Paris is the capital of France .", "tags": ["Paris", "paris"]},
    ...          {"document": "It is known as the city of lights .", "tags": ["lights", "paris"]},
    ...          {"document": "The Eiffel Tower can be found in Paris .", "tags": ["Eiffel", "paris"]},
    ...      ]),
    ...     ("Toulouse", [
    ...          {"document": "Toulouse is the capital of Occitanie .", "tags": ["Toulouse", "Occitanie"]},
    ...          {"document": "It is known as the pink city .", "tags": ["Toulouse", "pink", "rose"]},
    ...          {"document": "Toulouse has a famous rugby club .", "tags": ["Toulouse", "rugby"]},
    ...      ]),
    ... ]

    >>> retriever = retrieve.Flash(on="tags")

    >>> ranker = rank.Encoder(
    ...    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    ...    on = "document",
    ...    k = 3,
    ...    path = "encoder.pkl"
    ... )

    >>> search = retriever + ranker

    >>> search = search.add(documents)

    >>> eval.eval(search=search, query_answers=query_answers, on="document")
    {'Precision@1': '100.00%', 'Precision@2': '100.00%', 'Precision@3': '100.00%', 'R-Precision': '83.33%', 'Precision': '100.00%'}

    """
    precision = collections.defaultdict(lambda: stats.Mean())
    global_precision = stats.Mean()
    r_precision = stats.Mean()

    for q, answers in query_answers:
        documents = search(q=q)

        answers = [answer[on] for answer in answers]
        documents = [document[on] for document in documents]

        # Precision @ k
        for k, document in enumerate(documents):
            if document in answers:
                precision[k].update(1)
                global_precision.update(1)
            else:
                precision[k].update(0)
                global_precision.update(0)

        # R-Precision
        relevant = 0
        for document in documents[: len(answers)]:
            if document in answers:
                relevant += 1
        r_precision.update(relevant / len(answers) if relevant > 0 else 0)

    metrics = {f"Precision@{k + 1}": f"{metric.get():.2%}" for k, metric in precision.items()}
    metrics.update({"R-Precision": f"{r_precision.get():.2%}"})
    metrics.update({"Precision": f"{global_precision.get():.2%}"})
    return metrics
