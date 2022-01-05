import setuptools

from cherche.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cherche",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Another retriever, ranker, reader.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/cherche",
    packages=setuptools.find_packages(),
    install_requires=[
        "creme == 0.6.1",
        "elasticsearch == 7.10.0",
        "faiss-cpu == 1.7.1.post3",
        "flashtext == 2.7",
        "numpy == 1.19.0",
        "rank-bm25 == 0.2.1",
        "sentence-transformers==2.1.0",
        "tqdm == 4.62.3",
        "transformers == 4.12.0",
        "lunr == 0.6.1",
    ],
    package_data={"cherche": ["data/towns.json", "data/semanlink/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
