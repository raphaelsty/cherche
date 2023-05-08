import pytest

from .. import rank, retrieve


def cherche_retrievers(key: str, on: str):
    """List of retrievers available in cherche."""
    for retriever in [
        retrieve.TfIdf,
        retrieve.Lunr,
    ]:
        yield retriever(key=key, on=on, documents=documents())


def documents():
    return [
        {
            "id": 0,
            "title": "Paris",
            "article": "This town is the capital of France",
            "author": "Wikipedia",
        },
        {
            "id": 1,
            "title": "Eiffel tower",
            "article": "Eiffel tower is based in Paris",
            "author": "Wikipedia",
        },
        {
            "id": 2,
            "title": "Montreal",
            "article": "Montreal is in Canada.",
            "author": "Wikipedia",
        },
    ]


@pytest.mark.parametrize(
    "search, documents, k",
    [
        pytest.param(
            (retriever_a * retriever_b * retriever_c) + documents(),
            documents(),
            k,
            id=f"Union retrievers: {retriever_a.__class__.__name__} | {retriever_b.__class__.__name__} | {retriever_c.__class__.__name__} k: {k}",
        )
        for k in [None, 3, 4]
        for retriever_c in cherche_retrievers(key="id", on="title")
        for retriever_b in cherche_retrievers(key="id", on="article")
        for retriever_a in cherche_retrievers(key="id", on="author")
    ],
)
def test_retriever_union(search, documents: list, k: int):
    """Test retriever union operator."""
    # Empty documents
    search = search.add(documents)

    answers = search(q="France", k=k)
    assert len(answers) == min(k, 1) if k is not None else 1

    for sample in answers:
        for key in ["title", "article", "author"]:
            assert key in sample

    answers = search(q="Wikipedia", k=k)
    assert len(answers) == min(k, len(documents)) if k is not None else len(documents)

    answers = search(q="Unknown", k=k)
    assert len(answers) == 0
