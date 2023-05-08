import abc
import collections
import typing

__all__ = ["Compose"]


class Compose(abc.ABC):
    """Base class for Pipeline."""

    def __init__(self, models: typing.List) -> None:
        self.models = models
        for model in self.models:
            if hasattr(model, "key"):
                self.key = model.key
                break

    @staticmethod
    def _build_query(
        q: typing.Union[typing.List[str], str],
        batch_size: typing.Optional[int],
        k: typing.Optional[int],
        documents: typing.Optional[typing.List[typing.Dict[str, str]]] = None,
        **kwargs,
    ) -> typing.Dict[str, typing.Any]:
        """Build the query for the model."""
        if isinstance(q, str):
            q = [q]
            if documents is not None:
                documents = [documents]
        return {
            "batch_size": batch_size,
            "k": k,
            "documents": documents,
            "q": q,
            **kwargs,
        }

    def _build_match(self, query: typing.Dict[str, typing.Any]):
        match = collections.defaultdict(list)
        for model_id, model in enumerate(self.models):
            # Call the model
            retrieved = model(**query)
            if not retrieved:
                continue

            for n_query, documents in enumerate(retrieved):
                match[n_query].extend(documents)

        return match

    def _scores(
        self, match: typing.Dict[int, typing.List[typing.Dict[str, str]]]
    ) -> typing.Tuple[
        typing.List[typing.Dict[str, float]], typing.List[typing.Dict[str, float]]
    ]:
        """Compute scores for each document of the union."""
        queries_scores, queries_counter = [], []
        for documents_query in match.values():
            rank = collections.defaultdict(float)
            counter = collections.defaultdict(int)
            for r, document in enumerate(documents_query):
                rank[document[self.key]] += 1 / (r + 1)
                counter[document[self.key]] += 1
            queries_scores.append({key: counter[key] * rank[key] for key in counter})
            queries_counter.append(counter)
        return queries_scores, queries_counter

    @abc.abstractmethod
    def __call__(self, q: str, **kwargs) -> list:
        return []

    def add(self, documents: list, **kwargs) -> "Compose":
        """Add new documents."""
        history = {}
        for model in self.models:
            if hasattr(model, "add") and callable(model.add):
                # Avoid indexing twice the same model, index or store.
                if id(model) in history:
                    continue
                if hasattr(model, "index"):
                    if id(model.index) in history:
                        continue
                if hasattr(model, "store"):
                    if id(model.store) in history:
                        continue

                model = model.add(documents=documents, **kwargs)

                history[id(model)] = True
                if hasattr(model, "index"):
                    history[model.index] = True

                if hasattr(model, "store"):
                    history[model.store] = True

        return self

    def reset(self) -> "Compose":
        for model in self.models:
            if hasattr(model, "reset") and callable(model.reset):
                model = model.reset()
        return self

    def __repr__(self) -> str:
        repr = "\n".join(
            [
                model.__repr__()
                if not isinstance(model, dict)
                else "Mapping to documents"
                for model in self.models
            ]
        )
        return repr


def rank_union(
    key: str,
    match: typing.Dict[int, typing.List[typing.List[typing.Dict[str, str]]]],
    scores: typing.List[typing.Dict[str, float]],
) -> typing.List[typing.List[typing.Dict[str, str]]]:
    """Rank documents of the union."""
    queries_rank = []
    for documents_query, scores_query in zip(match.values(), scores):
        query_seen, query_rank = {}, []

        for document in documents_query:
            key_value = document[key]

            if key_value not in query_seen:
                # Remove similarity
                document.pop("similarity")

                # Append the document with it's new score
                query_rank.append({**document, "similarity": scores_query[key_value]})

                # Seen document
                query_seen[key_value] = True

        queries_rank.append(query_rank)

    return queries_rank


def rank_intersection(
    key: str,
    models: typing.List,
    match: typing.Dict[int, typing.List[typing.List[typing.Dict[str, str]]]],
    scores: typing.List[typing.Dict[str, float]],
    counter: typing.List[typing.Dict[str, float]],
) -> typing.List[typing.List[typing.Dict[str, str]]]:
    """Rank documents of the union."""
    queries_rank = []
    for documents_query, scores_query, counter_query in zip(
        match.values(), scores, counter
    ):
        query_seen, query_rank = {}, []

        for document in documents_query:
            key_value = document[key]

            if key_value not in query_seen and counter_query[key_value] == len(models):
                # Remove similarity
                document.pop("similarity")

                # Append the document with it's new score
                query_rank.append({**document, "similarity": scores_query[key_value]})

                # Seen document
                query_seen[key_value] = True

        queries_rank.append(query_rank)

    return queries_rank


def rank_vote(
    key: str,
    match: typing.Dict[int, typing.List[typing.List[typing.Dict[str, str]]]],
    scores: typing.List[typing.Dict[str, float]],
) -> typing.List[typing.List[typing.Dict[str, str]]]:
    """Rank documents of the union."""
    queries_rank = []
    for documents_query, scores_query in zip(match.values(), scores):
        query_rank = []
        index = {document[key]: document for document in documents_query}
        for key_value in sorted(scores_query, key=scores_query.get, reverse=True):
            document = index[key_value]
            document.pop("similarity")
            query_rank.append({**document, "similarity": scores_query[key_value]})
        queries_rank.append(query_rank)
    return queries_rank
