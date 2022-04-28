import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="weighted_nmf",
    version="0.0.1",
    author="Ruhi Doshi",
    author_email="rdoshi99@berkeley.edu",
    description="Variance-weighted non-negative matrix factorization package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdoshi99/weighted_nmf",
    packages=['weighted_nmf'],
    install_requires=['numpy', 'tqdm']
)
