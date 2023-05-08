__all__ = ["evaluation"]

import collections
import typing

import numpy as np

__all__ = ["evaluation"]

class Mean:
    """Online running mean.

    Reference
    ---------
    1. River [https://github.com/online-ml/river/blob/main/river/stats/mean.py]
    """

    def __init__(self):
        self.n = 0
        self._mean = 0.0

    def update(self, x, w=1.0):
        self.n += w
        self._mean += (w / self.n) * (x - self._mean)
        return self

    def get(self):
        return self._mean


def evaluation(
    search,
    query_answers: typing.List[typing.Tuple[str, typing.List[typing.Dict[str, str]]]],
    hits_k: range = range(1, 6, 1),
    batch_size: typing.Optional[int] = None,
    k: typing.Optional[int] = None,
):
    """Evaluation function


    Parameters
    ----------
    search
        Search function.
    query_answers
        List of tuples (query, answers).
    hits_k
        List of k to compute precision, recall and F1.
    k
        Number of documents to retrieve.
    batch_size
        Batch size.

    Examples
    --------
    >>> from pprint import pprint as print
    >>> from cherche import data, evaluate, retrieve
    >>> from sklearn.feature_extraction.text import TfidfVectorizer

    >>> documents, query_answers = data.arxiv_tags(
    ...    arxiv_title=True, arxiv_summary=False, comment=False
    ... )

    >>> search = retrieve.TfIdf(
    ...     key="uri",
    ...     on=["prefLabel_text", "altLabel_text"],
    ...     documents=documents,
    ...     tfidf=TfidfVectorizer(lowercase=True, ngram_range=(3, 7), analyzer="char"),
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

    """
    precision = collections.defaultdict(lambda: Mean())
    recall = collections.defaultdict(lambda: Mean())
    f1 = collections.defaultdict(lambda: Mean())
    r_precision = Mean()

    answers = search(
        **{
            "q": [q for q, _ in query_answers],
            "batch_size": batch_size,
            "k": k,
        }
    )

    for (q, golds), candidates in zip(query_answers, answers):
        candidates = [candidate[search.key] for candidate in candidates]
        golds = {gold[search.key]: True for gold in golds}

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
            recall[k].update(positives / len(golds)) if positives > 0 else recall[
                k
            ].update(0)

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
            (2 * precision[k].get() * recall[k].get())
            / (precision[k].get() + recall[k].get())
            if (precision[k].get() + recall[k].get()) > 0
            else 0
        )

    metrics = {
        f"Precision@{k}": f"{metric.get():.2%}" for k, metric in precision.items()
    }
    metrics.update(
        {f"Recall@{k}": f"{metric.get():.2%}" for k, metric in recall.items()}
    )
    metrics.update({f"F1@{k}": f"{metric:.2%}" for k, metric in f1.items()})
    metrics.update({"R-Precision": f"{r_precision.get():.2%}"})
    return metrics
