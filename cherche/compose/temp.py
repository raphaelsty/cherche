from cherche import rank, retrieve
from sentence_transformers import SentenceTransformer

documents = [
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

ranker = rank.Encoder(
    encoder=SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    key="id",
    on=["title", "article"],
    k=3,
)

a = retrieve.TfIdf(key="id", on=["title", "article"], k=3, documents=documents) + ranker
b = retrieve.Flash(key="id", on=["title", "article"], k=3) + ranker
b.add(documents)


print((a | b)("paris montreal eiffel"))
