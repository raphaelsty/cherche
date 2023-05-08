import pytest

from .. import retrieve


def cherche_retrievers(on: str):
    """List of retrievers available in cherche."""
    yield from [
        retrieve.TfIdf(key="title", on=on, documents=documents()),
        retrieve.Lunr(key="title", on=on, documents=documents()),
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


@pytest.mark.parametrize(
    "retriever, documents, k",
    [
        pytest.param(
            retriever,
            documents(),
            k,
            id=f"retriever: {retriever.__class__.__name__}, k: {k}",
        )
        for k in [None, 2, 4]
        for retriever in cherche_retrievers(on="article")
    ],
)
def test_retriever(retriever, documents: list, k: int):
    """Test retriever. Test if the number of retrieved documents is coherent.
    Check for unknown tokens in the corpus, should returns an empty list.
    """
    retriever = retriever + documents
    retriever.add(documents)

    # A single document contains town.
    answers = retriever(q="town", k=k)
    if k is None or k >= 1:
        assert len(answers) >= 1
    else:
        assert len(answers) == 0

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="un", k=k)
    assert len(answers) == 0

    # All documents contains "Montreal Eiffel France"
    answers = retriever(q="Montreal Eiffel France", k=k)
    if k is None or k >= len(documents):
        assert len(answers) == len(documents)
    else:
        assert len(answers) == k


@pytest.mark.parametrize(
    "retriever, documents, k",
    [
        pytest.param(
            retriever,
            documents(),
            k,
            id=f"Multiple fields retriever: {retriever.__class__.__name__}, k: {k}",
        )
        for k in [None, 2, 4]
        for retriever in cherche_retrievers(on=["article", "title", "author"])
    ],
)
def test_fields_retriever(retriever, documents: list, k: int):
    """Test retriever when providing multiples fields."""
    retriever = retriever + documents

    # All documents have Wikipedia as author.
    answers = retriever(q="Wikipedia", k=k)
    if k is None or k >= len(documents):
        assert len(answers) == len(documents)
    else:
        assert len(answers) == max(k, 0)

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="un")
    assert len(answers) == 0

    # Two documents contains paris
    answers = retriever(q="Paris", k=k)

    if k is None or k >= 2:
        assert len(answers) == 2
    else:
        assert len(answers) == max(k, 0)


@pytest.mark.parametrize(
    "documents, k",
    [
        pytest.param(
            documents(),
            k,
            id=f"retriever: Flash, k: {k}",
        )
        for k in [None, 2, 4]
    ],
)
def test_flash(documents: list, k: int):
    """Test Flash retriever."""
    # Reset retriever
    retriever = retrieve.Flash(key="title", on="title") + documents
    retriever.add(documents)

    # A single document contains town.
    answers = retriever(q="paris", k=k)
    if k is None or k >= 1:
        assert len(answers) == 1
    else:
        assert len(answers) == 0

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    # Unknown token.
    answers = retriever(q="Unknown", k=k)
    assert len(answers) == 0

    # All documents contains is
    answers = retriever(q="Paris Eiffel tower Montreal", k=k)

    if k is None or k >= len(documents):
        assert len(answers) == len(documents)
    else:
        assert len(answers) == k
