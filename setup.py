import setuptools

from cherche.__version__ import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = [
    "numpy >= 1.24.3",
    "scikit-learn >= 1.2.2",
    "lunr >= 0.6.2",
    "rapidfuzz >= 3.0.0",
    "flashtext >= 2.7",
    "tqdm >= 4.62.3",
    "scipy >= 1.7.3",
]

cpu = ["sentence-transformers >= 2.2.2", "faiss-cpu >= 1.7.4"]
gpu = ["sentence-transformers >= 2.2.2", "faiss-gpu >= 1.7.4"]
dev = [
    "numpydoc >= 1.4.0",
    "mkdocs_material >= 8.3.5",
    "mkdocs-awesome-pages-plugin >= 2.7.0",
    "mkdocs-jupyter >= 0.21.0",
    "pytest-cov >= 4.0.0",
    "pytest >= 7.3.1",
    "isort >= 5.12.0",
    "ipywidgets >= 8.0.6",
]

setuptools.setup(
    name="cherche",
    version=f"{__version__}",
    license="MIT",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    description="Neural Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/cherche",
    download_url="https://github.com/user/cherche/archive/v_01.tar.gz",
    keywords=[
        "neural search",
        "information retrieval",
        "question answering",
        "semantic search",
    ],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    extras_require={
        "cpu": base_packages + cpu,
        "gpu": base_packages + gpu,
        "dev": base_packages + cpu + dev,
    },
    package_data={"cherche": ["data/towns.json", "data/semanlink/*.json"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
