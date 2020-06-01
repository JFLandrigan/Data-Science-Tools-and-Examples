from setuptools import setup, find_packages

setup(
    name="Model_Tuner",
    version="0.0.1",
    description="Machine Learning Package to help practitioners develop models",
    url="",
    author="Jon-Frederick Landrigan",
    author_email="jon.landrigan@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "nltk",
        "scikit-learn",
        "matplotlib",
        "string",
        "gensim",
        "time",
        "scikit-optimize",
    ],
    zip_safe=False,
)
