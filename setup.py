from setuptools import setup, find_packages


setup(
    name="kaa-xai",
    version="25.12.0",
    url="https://github.com/IRT-Saint-Exupery/KAA",
    author="Philippe Dejean",
    author_email="philippe.dejean@irt-saintexupery.com",
    description=("KAA is a framework allowing to apply several explainability methods and metrics"
                 "from several dedicated libraries on AI model to verify them."
                ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    readme="README.md",
    install_requires=[],
    python_requires="==3.8",
    classifiers=["Programming Language :: Python :: 3.8",
                 "License :: OSI Approved :: Apache Software License",
                 "Operating System :: OS Independent"],
)
