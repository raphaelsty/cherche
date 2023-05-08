import pytest

from .. import rank


def cherche_rankers(key: str, on: str):
    """List of rankers available in cherche."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    yield from [
        rank.DPR(
            key=key,
            on=on,
            encoder=SentenceTransformer(
                "facebook-dpr-ctx_encoder-single-nq-base"
            ).encode,
            query_encoder=SentenceTransformer(
                "facebook-dpr-question_encoder-single-nq-base"
            ).encode,
        ),
        rank.Encoder(
            key=key,
            on=on,
            encoder=SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2"
            ).encode,
        ),
        rank.CrossEncoder(
            on=on,
            encoder=CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1").predict,
        ),
    ]


def documents():
    return [
        {
            "title": "Paris",
            "article": "This town is the capital of France",
            "author": "Wikipedia",
        },
        {
            "title": "Eiffel tower",
            "article": "Eiffel tower is based in Paris",
            "author": "Wikipedia",
        },
        {
            "title": "Montreal",
            "article": "Montreal is in Canada.",
            "author": "Wikipedia",
        },
    ]


def missing_documents():
    return []


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
        for k in [None, 2, 4]
        for ranker in cherche_rankers(key="title", on="article")
    ],
)
def test_ranker(ranker, documents: list, key: str, k: int):
    """Test ranker. Test if the number of ranked documents is coherent.
    Check for empty retrieved documents should returns an empty list.
    """
    if not isinstance(ranker, rank.CrossEncoder):
        ranker.add(documents)

    # CrossEncoder shot needs all the fields
    if not isinstance(ranker, rank.CrossEncoder):
        ranker += documents
        # Convert inputs document to a list of id [{"id": 0}, {"id": 1}, {"id": 2}]
        documents = [{key: document[key]} for document in documents]

    answers = ranker(q="Eiffel tower France", documents=documents, k=k)

    if k is not None:
        assert len(answers) == min(k, len(documents))
    else:
        assert len(answers) == len(documents)

    for index, sample in enumerate(answers):
        for key in ["title", "article", "author"]:
            assert key in sample

        if index == 0:
            assert sample["title"] == "Eiffel tower"

    answers = ranker(q="Canada", documents=documents, k=k)

    if k is None:
        assert answers[0]["title"] == "Montreal"
    elif k >= 1:
        assert answers[0]["title"] == "Montreal"
    else:
        assert len(answers) == 0

    # Unknown token.
    answers = ranker(q="Paris", documents=[], k=k)
    assert len(answers) == 0


@pytest.mark.parametrize(
    "ranker, documents, key, k",
    [
        pytest.param(
            ranker,
            missing_documents(),
            "title",
            5,
            id=f"Ranker: {ranker.__class__.__name__}, missing documents, k: 5",
        )
        for ranker in cherche_rankers(key="title", on="article")
    ],
)
def test_ranker_missing_documents(ranker, documents: list, key: str, k: int):
    """Test ranker when retriever do not returns any documents."""
    answers = ranker(q="Eiffel tower France", documents=documents, k=k)
    assert len(answers) == 0

    answers = ranker(
        q=["Eiffel tower France", "Montreal Canada"],
        documents=[documents, documents],
        k=k,
    )
    assert len(answers) == 2 and len(answers[0]) == 0 and len(answers[1]) == 0
