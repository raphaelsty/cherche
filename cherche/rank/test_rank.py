import pytest
from cherche.rank.zero_shot import ZeroShot

from .. import rank


def cherche_rankers(key: str, on: str, k: int = None, path: str = None):
    """List of rankers available in cherche."""
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline

    yield from [
        rank.DPR(
            key=key,
            on=on,
            encoder=SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base").encode,
            query_encoder=SentenceTransformer(
                "facebook-dpr-question_encoder-single-nq-base"
            ).encode,
            k=k,
            path=path,
        ),
        rank.Encoder(
            key=key,
            on=on,
            encoder=SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
            k=k,
            path=path,
        ),
        rank.ZeroShot(
            key=key,
            on=on,
            encoder=pipeline(
                "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"
            ),
            k=k,
        ),
    ]


def documents():
    return [
        {"title": "Paris", "article": "This town is the capital of France", "author": "Wikipedia"},
        {
            "title": "Eiffel tower",
            "article": "Eiffel tower is based in Paris",
            "author": "Wikipedia",
        },
        {"title": "Montreal", "article": "Montreal is in Canada.", "author": "Wikipedia"},
    ]


@pytest.mark.parametrize(
    "ranker, documents, key, k",
    [
        pytest.param(
            ranker,
            documents(),
            "title",
            k,
            id=f"Ranker: {ranker.__class__.__name__}, k: {k}",
        )
        for k in [None, 0, 2, 4]
        for ranker in cherche_rankers(key="title", on="article", k=k)
    ],
)
def test_ranker(ranker, documents: list, key: str, k: int):
    """Test ranker. Test if the number of ranked documents is coherent.
    Check for empty retrieved documents should returns an empty list.
    """
    ranker.add(documents)

    # Zero shot needs all the fields
    if not isinstance(ranker, ZeroShot):
        ranker += documents
        # Convert inputs document to a list of id [{"id": 0}, {"id": 1}, {"id": 2}]
        documents = [{key: document[key]} for document in documents]

    answers = ranker(q="Eiffel tower France", documents=documents)

    if k is not None:
        assert len(answers) == min(k, len(documents))
    else:
        assert len(answers) == len(documents)

    for index, sample in enumerate(answers):

        for key in ["title", "article", "author"]:
            assert key in sample

        if index == 0:
            assert sample["title"] == "Eiffel tower"

    answers = ranker(q="Canada", documents=documents)

    if k is None:
        assert answers[0]["title"] == "Montreal"
    elif k >= 1:
        assert answers[0]["title"] == "Montreal"
    else:
        assert len(answers) == 0

    # Unknown token.
    answers = ranker(q="Paris", documents=[])
    assert len(answers) == 0
